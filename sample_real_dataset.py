import argparse
import os

from torchvision.utils import save_image

from utils import get_train_set

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--output", type=str, default="sampled_images/fashion_real.png")
    parser.add_argument("--dataset", type=str, default="FashionMNIST")

    args = parser.parse_args()

    output_path = args.output
    nrow = args.nrow
    num_samples = nrow * nrow

    dataloader = get_train_set(num_samples, args.dataset)

    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    selected_images = images[:num_samples]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_image(selected_images, output_path, nrow=nrow)
