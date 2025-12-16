import sys
import os


from loguru import logger
import torch

from model_mnist import CondRefineNetDilated
from trainer import NCSNTrainer
from utils import config, get_train_set, device

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    args = parser.parse_args()

    checkpoint_path = os.path.join("checkpoints", args.checkpoint_name)
    samples_path = os.path.join("sampled_images", args.samples_name)

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
        0,
        args.beta1,
        args.beta2,
        checkpoint_path,
        1,
    )

    logger.info(f"Loading model from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    trained_ncsn = CondRefineNetDilated(mnist_config).to(device)
    trained_ncsn.load_state_dict(state_dict)
    trained_ncsn.eval()

    eps = 2e-5
    T = 100
    nrow = 4

    logger.info(f"Saving samples to {samples_path}")
    trainer.save_sample_grid(nrow, trained_ncsn, eps, T, output_path=samples_path)
