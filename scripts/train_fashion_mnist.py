import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model_mnist_plus import CondRefineNetDilated
from trainer import NCSNTrainer
from utils import get_train_set, config, device


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_iters", type=int, default=100001)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--logit_transform", action="store_true", default=False)
    parser.add_argument("--random_flip", action="store_true", default=True)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--ngf", type=int, default=96)
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--ema_rate", type=float, default=0.999)
    parser.add_argument("--checkpoints_folder", type=str, default="checkpoints")
    parser.add_argument("--samples_folder", type=str, default="sampled_images")
    args = parser.parse_args()

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
    trainer.train_ncsn()

    # Sampling test
    saved_idx = int((args.num_iters - 1) // args.save_every)
    if saved_idx > 0:
        model_name = f"fashion_{saved_idx}.pt"
        model_path = os.path.join(args.checkpoints_folder, model_name)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            trained_ncsn = CondRefineNetDilated(fashion_mnist_config).to(device)
            trained_ncsn.load_state_dict(state_dict)
            trained_ncsn.eval()

            eps = 2e-5
            T = 100
            nrow = 4
            output_path = os.path.join(
                args.samples_folder, f"fashion_{saved_idx * args.save_every}.png"
            )
            trainer.save_sample_grid(
                nrow, trained_ncsn, eps, T, output_path=output_path
            )
