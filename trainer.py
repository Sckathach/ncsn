import os
from math import sqrt

import torch
from loguru import logger
from rich.progress import track
from torch.optim import Adam
from torchvision.utils import save_image

from utils import device


class NCSNTrainer:
    def __init__(
        self,
        ncsn,
        lr: float,
        dataloader,
        num_iters: int,
        beta1: float,
        beta2: float,
        sigma_start: float,
        sigma_end: float,
        n_sigmas: int,
        checkpoints_folder: str,
        save_every: int = 5000,
        weight_decay: float = 0.01,
        ema_rate: float = 0.999,
        checkpoint_name_prefix: str = "ncsn",
    ):
        # compute sigmas as sigma_1 / sigma_2 = ... = sigma_(n-1) / sigma_n = gamma
        gamma = (sigma_start / sigma_end) ** (1 / n_sigmas)
        self.sigmas = torch.Tensor(
            [sigma_start * gamma**i for i in range(0, n_sigmas)]
        ).to(device)

        self.ncsn = ncsn
        self.ncsn_opt = Adam(
            self.ncsn.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            weight_decay=weight_decay,  # adding weight decays to slow down memorization
        )

        self.dataloader = dataloader
        self.num_iters = num_iters
        self.save_folder = checkpoints_folder
        self.save_every = save_every
        self.checkpoint_name_prefix = checkpoint_name_prefix

        # Adding EMA to slow down learning and collapsing
        self.ema_rate = ema_rate
        if self.ema_rate > 0:
            self.ema_ncsn = type(ncsn)(ncsn.config).to(device)
            self.ema_ncsn.load_state_dict(ncsn.state_dict())
            self.ema_ncsn.eval()
        else:
            self.ema_ncsn = None

    def _loss_tensor(self, batch, which_sigmas):
        batch_size = batch.shape[0]

        # Compute target
        selected_sigmas = torch.index_select(self.sigmas, 0, which_sigmas)
        selected_sigmas = selected_sigmas.view(
            [batch.shape[0]] + [1] * (len(batch.shape) - 1)
        )
        perturbed_batch = batch + torch.randn_like(batch) * selected_sigmas
        target = -(perturbed_batch - batch) / (selected_sigmas**2)

        # Compute output
        output = self.ncsn(perturbed_batch, which_sigmas)

        # Loss is euclidean norm squared of the difference
        diff = target - output
        normsq = torch.sum(diff**2, dim=[i for i in range(1, len(diff.shape))])
        loss = (1 / (2 * batch_size)) * normsq * torch.squeeze(selected_sigmas) ** 2
        return loss.sum()

    def _update_ema(self):
        if self.ema_ncsn is not None:
            for ema_param, param in zip(
                self.ema_ncsn.parameters(), self.ncsn.parameters()
            ):
                ema_param.data.mul_(self.ema_rate).add_(
                    param.data, alpha=1 - self.ema_rate
                )

    def train_ncsn(self):
        os.makedirs(self.save_folder, exist_ok=True)
        curr_iter = 0
        while True:
            for batch, __ in self.dataloader:
                batch = batch.to(device)
                self.ncsn_opt.zero_grad()

                # Choose random sigma for each image in batch to minimize expected loss
                which_sigmas = torch.randint(
                    0, len(self.sigmas), (batch.shape[0],), device=device
                )
                loss = self._loss_tensor(batch, which_sigmas)
                loss.backward()
                self.ncsn_opt.step()
                self._update_ema()

                if curr_iter % 10 == 0:
                    logger.info(f"iter: {curr_iter}  |  loss: {round(loss.item(), 3)}")
                curr_iter += 1
                if curr_iter % self.save_every == 0 and curr_iter > 0:
                    self._save_model(curr_iter)
                if curr_iter >= self.num_iters:
                    return

    def _save_model(self, curr_iter):
        save_num = int(curr_iter)
        filename = f"{self.checkpoint_name_prefix}_{save_num}.pt"
        save_path = os.path.join(self.save_folder, filename)
        torch.save(self.ncsn.state_dict(), save_path)
        if self.ema_ncsn is not None:
            filename_ema = f"{self.checkpoint_name_prefix}_ema_{save_num}.pt"
            save_path_ema = os.path.join(self.save_folder, filename_ema)
            torch.save(self.ema_ncsn.state_dict(), save_path_ema)

    def _get_image_shape(self):
        batch, __ = next(iter(self.dataloader))
        shape = batch.shape[1:]
        return shape

    def annealed_langevin_dynamics(self, trained_ncsn, eps, T, num_samples):
        image_shape = self._get_image_shape()
        x_prev = torch.rand(num_samples, *image_shape).to(
            device
        )  # uniform noise as starting point

        for i in track(range(len(self.sigmas))):
            logger.info(f"Starting sigma {i}")
            ai = eps * self.sigmas[i] ** 2 / self.sigmas[-1] ** 2
            i_tensor = torch.Tensor([i] * num_samples).long().to(device)

            for _ in track(range(T)):
                zt = torch.randn_like(x_prev)  # normal 0, I
                score = trained_ncsn(x_prev, i_tensor)
                x_curr = x_prev + score * ai / 2 + sqrt(ai) * zt
                x_prev = x_curr

            logger.info(f"finished sigma_{i}")

        return x_curr

    def save_sample_grid(
        self, nrow, trained_ncsn, eps, T, output_path="sampled_images/samples.png"
    ):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        num_samples = nrow**2
        with torch.no_grad():
            samples = self.annealed_langevin_dynamics(trained_ncsn, eps, T, num_samples)
        save_image(samples, output_path, nrow=nrow)
