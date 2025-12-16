import sys
import os


from loguru import logger
import argparse
from torchvision.utils import save_image

from utils import get_train_set

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample real images from the Fashion MNIST dataset."
    )
    parser.add_argument(
        "--nrow",
        type=int,
        required=True,
        help="Number of rows and columns for the sample grid.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sampled_images/fashion_real.png",
        help="Path to save the generated image.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="FashionMNIST",
        help="Dataset to sample from.",
    )

    args = parser.parse_args()

    output_path = args.output
    nrow = args.nrow
    num_samples = nrow * nrow

    dataloader = get_train_set(num_samples, args.dataset)

    logger.info(f"Fetching {num_samples} real images from {args.dataset}...")

    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    selected_images = images[:num_samples]

    logger.info(f"Saving {nrow}x{nrow} samples to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(selected_images, output_path, nrow=nrow)
    logger.info("Done.")
