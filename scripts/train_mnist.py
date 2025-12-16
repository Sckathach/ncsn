import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from loguru import logger
import torch

from model_mnist import CondRefineNetDilated
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
    parser.add_argument("--random_flip", action="store_true", default=False)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--checkpoints_folder", type=str, default="../checkpoints")
    parser.add_argument("--samples_name", type=str, default="samples.png")
    parser.add_argument("--save_every", type=int, default=10000)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    mnist_config = config(
        "MNIST",
        args.image_size,
        args.channels,
        args.logit_transform,
        args.random_flip,
        args.num_classes,
        args.ngf,
    )

    ncsn = CondRefineNetDilated(mnist_config).to(device)

    dataloader = get_train_set(args.batch_size, "MNIST")

    trainer = NCSNTrainer(
        ncsn,
        args.lr,
        dataloader,
        args.num_iters,
        args.beta1,
        args.beta2,
        args.checkpoints_folder,
        args.save_every,
        checkpoint_name_prefix="mnist",
    )
    trainer.train_ncsn()

    state_dict = torch.load(args.checkpoint, map_location=device)
    trained_ncsn = CondRefineNetDilated(mnist_config).to(device)
    trained_ncsn.load_state_dict(state_dict)
    trained_ncsn.eval()

    eps = 2e-5
    T = 100
    nrow = 4
    output_path = os.path.join(args.samples_folder, args.samples_name)
    trainer.save_sample_grid(nrow, trained_ncsn, eps, T, output_path=output_path)
