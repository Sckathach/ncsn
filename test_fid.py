import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from rich.progress import track
from scipy import linalg
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from large_unet import CondRefineNetDilated
from utils import config


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    if mu1.shape != mu2.shape:
        raise ValueError(
            f"Training and test mean vectors have different lengths: {mu1.shape} != {mu2.shape}"
        )
    if sigma1.shape != sigma2.shape:
        raise ValueError(
            f"Training and test covariances have different dimensions: {sigma1.shape} != {sigma2.shape}"
        )

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        logger.warning(
            f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates"
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            logger.warning(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class InceptionV3FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.block3 = nn.Sequential(
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)


def get_activations(loader, model, dims=2048, device="cuda"):
    model.eval()
    pred_arr = []

    for batch in track(loader, description="Calculating activations"):
        if isinstance(batch, list) or isinstance(batch, tuple):
            batch = batch[0]

        batch = batch.to(device)

        if batch.shape[1] == 1:
            batch = batch.repeat(1, 3, 1, 1)

        batch = F.interpolate(
            batch, size=(299, 299), mode="bilinear", align_corners=False
        )

        with torch.no_grad():
            pred = model(batch)

        pred_arr.append(pred.cpu().data.numpy())

    pred_arr = np.concatenate(pred_arr, axis=0)
    return pred_arr


def calculate_activation_statistics(loader, model, dims=2048, device="cuda"):
    act = get_activations(loader, model, dims, device)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def generate_samples(ncsn, num_samples, batch_size, device):
    sigmas = torch.Tensor([0.599**i for i in range(0, 10)]).to(device)
    eps_val = 2e-5
    T = 100

    ncsn.eval()
    all_samples = []

    num_batches = (num_samples + batch_size - 1) // batch_size

    logger.info(f"Generating {num_samples} samples...")
    for _ in track(range(num_batches), description="Sampling batches"):
        current_batch_size = min(batch_size, num_samples - len(all_samples))
        if current_batch_size <= 0:
            break

        x = torch.rand(current_batch_size, 1, 32, 32).to(device)

        with torch.no_grad():
            for i, sigma in enumerate(sigmas):
                alpha = eps_val * (sigma**2) / (sigmas[-1] ** 2)
                labels = torch.full(
                    (current_batch_size,), i, dtype=torch.long, device=device
                )

                for t in range(T):
                    z = torch.randn_like(x)
                    score = ncsn(x, labels)
                    x = x + score * alpha / 2 + torch.sqrt(alpha) * z

        all_samples.append(x.cpu())

    return torch.cat(all_samples, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="MNIST", choices=["MNIST", "FashionMNIST"]
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--data_root", type=str, default="data")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # 1. Load Model
    logger.info(f"Loading model from {args.checkpoint}...")
    conf = config(
        dataset=args.dataset,
        image_size=32,
        channels=1,
        logit_transform=False,
        random_flip=False,
        num_classes=10,
        ngf=64,
    )

    model = CondRefineNetDilated(conf).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 2. Generate Samples
    samples = generate_samples(
        model, args.num_samples, args.batch_size, device
    )  # (N, 1, 32, 32) float

    # 3. Load Real Data
    logger.info(f"Loading {args.dataset} Test Set from {args.data_root}...")
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])

    if args.dataset == "MNIST":
        test_dataset = datasets.MNIST(
            args.data_root, train=False, download=True, transform=transform
        )
    else:
        test_dataset = datasets.FashionMNIST(
            args.data_root, train=False, download=True, transform=transform
        )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # 4. Inception Model
    logger.info("Loading InceptionV3...")
    inception = InceptionV3FeatureExtractor().to(device)
    inception.eval()

    # 5. Compute FID
    logger.info("Computing statistics for Real Images...")
    mu_real, sigma_real = calculate_activation_statistics(
        test_loader, inception, device=device
    )

    logger.info("Computing statistics for Generated Images...")

    class TensorDataset(torch.utils.data.Dataset):
        def __init__(self, tensor):
            self.tensor = tensor

        def __getitem__(self, index):
            return self.tensor[index]

        def __len__(self):
            return self.tensor.size(0)

    gen_loader = DataLoader(
        TensorDataset(samples), batch_size=args.batch_size, shuffle=False
    )
    mu_gen, sigma_gen = calculate_activation_statistics(
        gen_loader, inception, device=device
    )

    logger.info("Calculating FID Score...")
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_gen, sigma_gen)

    logger.info(f"FID Score ({args.dataset}): {fid_score}")


if __name__ == "__main__":
    main()
