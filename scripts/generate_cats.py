import torch
import math
from rich.progress import track

from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, utils
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
)

full = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
cat_idx = [i for i, (_, y) in enumerate(full) if y == 3]
ds = Subset(full, cat_idx)
loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=4, drop_last=True)

# Noise schedule (NCSN config)
L = 200
sigma_min, sigma_max = 0.01, 50.0
sigmas = torch.exp(torch.linspace(math.log(sigma_max), math.log(sigma_min), L)).to(
    device
)


# Model
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CondRCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.act = Swish()
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.dense = nn.Linear(time_emb_dim, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))
        # time embedding
        h += self.dense(t_emb)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.reshape(B, C, -1).permute(0, 2, 1)  # B, HW, C
        k = k.reshape(B, C, -1)  # B, C, HW
        v = v.reshape(B, C, -1).permute(0, 2, 1)  # B, HW, C

        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        w = torch.softmax(w, dim=-1)

        h = torch.bmm(w, v)  # B, HW, C
        h = h.permute(0, 2, 1).reshape(B, C, H, W)
        h = self.proj(h)
        return x + h


class ResNetScoreNetwork(nn.Module):
    def __init__(self, channels=[64, 128, 256]):
        super().__init__()
        self.time_emb_dim = 128

        # time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.time_emb_dim),
            Swish(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
        )

        self.conv_in = nn.Conv2d(3, channels[0], 3, padding=1)

        self.down1_0 = CondRCBlock(channels[0], channels[0], self.time_emb_dim)  # 64
        self.pool1 = nn.Conv2d(channels[0], channels[0], 3, stride=2, padding=1)
        self.down2_0 = CondRCBlock(channels[0], channels[1], self.time_emb_dim)
        self.pool2 = nn.Conv2d(channels[1], channels[1], 3, stride=2, padding=1)
        self.down3_0 = CondRCBlock(channels[1], channels[2], self.time_emb_dim)

        self.mid_0 = CondRCBlock(channels[2], channels[2], self.time_emb_dim)
        self.mid_1 = CondRCBlock(channels[2], channels[2], self.time_emb_dim)

        self.up_sample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up3_0 = CondRCBlock(
            channels[2] + channels[1], channels[1], self.time_emb_dim
        )

        self.up_sample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up2_0 = CondRCBlock(
            channels[1] + channels[0], channels[0], self.time_emb_dim
        )

        self.conv_out = nn.Conv2d(channels[0], 3, 3, padding=1)

    def forward(self, x, sigma):
        t = sigma.log().view(-1, 1)
        t_emb = self.time_mlp(t)
        x = self.conv_in(x)
        h1 = self.down1_0(x, t_emb)
        h = self.pool1(h1)
        h2 = self.down2_0(h, t_emb)
        h = self.pool2(h2)
        h3 = self.down3_0(h, t_emb)
        h = self.mid_0(h3, t_emb)
        h = self.mid_1(h, t_emb)
        h = self.up_sample1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up3_0(h, t_emb)
        h = self.up_sample2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up2_0(h, t_emb)
        out = self.conv_out(h)
        return out


class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


model = ResNetScoreNetwork().to(device)
opt = torch.optim.Adam(model.parameters(), lr=2e-4)

ema = EMA(model)


# Langevin
@torch.no_grad()
def sample_cats(n=25, T=100, eps=2e-5):
    # Load EMA weight
    ema.apply_shadow()
    model.eval()

    x = torch.randn(n, 3, 32, 32, device=device) * sigma_max

    for i, sigma in enumerate(track(sigmas, description="Sampling sigmas")):
        alpha = eps * (sigma / sigma_min) ** 2

        for t in range(T):
            z = torch.randn_like(x)

            pred_noise = model(x, sigma.expand(n))
            score = pred_noise / sigma

            x = x + 0.5 * alpha * score + torch.sqrt(alpha) * z

    x = x.clamp(0, 1)

    ema.restore()
    model.train()
    return x


steps = 40000
itr = iter(loader)

scaler = torch.amp.GradScaler("cuda")

print("Starting training...")
for step in track(range(steps)):
    try:
        x, _ = next(itr)
    except StopIteration:
        itr = iter(loader)
        x, _ = next(itr)

    x = x.to(device)
    idx = torch.randint(0, L, (x.size(0),), device=device)
    sigma = sigmas[idx].view(-1, 1, 1, 1)

    noise = torch.randn_like(x)
    xt = x + noise * sigma

    with torch.amp.autocast("cuda"):
        pred_noise = model(xt, sigmas[idx])
        # Loss: || pred_noise + noise ||^2
        target = -noise
        loss = ((pred_noise - target) ** 2).mean()

    opt.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    ema.update()

    if step % 500 == 0:
        print(f"{step}: {loss.item():.4f}")
        os.makedirs("check", exist_ok=True)

        # Save EMA
        ema.apply_shadow()
        torch.save(model.state_dict(), "check/cat_model_ema.pt")
        ema.restore()

    if step > 20000 and step % 10000 == 0:
        print(f"Sampling at step {step}...")
        os.makedirs("outputs", exist_ok=True)
        samples = sample_cats(n=25)
        utils.save_image(samples, f"outputs/cats_generated_{step}.png", nrow=5)


samples = sample_cats(n=25)
utils.save_image(samples, "outputs/cats_generated.png", nrow=5)
print("Saved: outputs/cats_generated.png")
