import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
import torch

from model_mnist_plus import CondRefineNetDilated
from trainer import NCSNTrainer
from utils import config, get_train_set, device


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--logit_transform", action="store_true", default=False)
    parser.add_argument("--random_flip", action="store_true", default=False)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--checkpoint_name", type=str, default="mnist_1.pt")
    parser.add_argument("--samples_name", type=str, default="mnist_sample.png")
    parser.add_argument("--num_iters", type=int, default=100001)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_rate", type=float, default=0.999)
    parser.add_argument("--checkpoints_folder", type=str, default="checkpoints")
    parser.add_argument("--samples_folder", type=str, default="sampled_images")
    args = parser.parse_args()

    checkpoint_path = os.path.join("checkpoints", args.checkpoint_name)
    samples_path = os.path.join("sampled_images", args.samples_name)
    fashion_mnist_config = config(
        "FashionMNIST",
        args.image_size,
        args.channels,
        args.logit_transform,
        args.random_flip,
        args.num_classes,
        args.ngf,
        args.dropout,
    )

    ncsn = CondRefineNetDilated(fashion_mnist_config).to(device)

    dataloader = get_train_set(args.batch_size, "FashionMNIST", args.random_flip)

    trainer = NCSNTrainer(
        ncsn,
        args.lr,
        dataloader,
        args.num_iters,
        args.beta1,
        args.beta2,
        args.checkpoints_folder,
        args.save_every,
        args.weight_decay,
        args.ema_rate,
        checkpoint_name_prefix="fashion",
    )

    logger.info(f"Loading model from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    trained_ncsn = CondRefineNetDilated(fashion_mnist_config).to(device)
    trained_ncsn.load_state_dict(state_dict)
    trained_ncsn.eval()

    eps = 2e-5
    T = 100
    nrow = 4

    logger.info(f"Saving samples to {samples_path}")
    trainer.save_sample_grid(nrow, trained_ncsn, eps, T, output_path=samples_path)
