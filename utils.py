import torchvision
import torch
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_train_set(batch_size, dataset_name="MNIST"):
    """
    Returns a DataLoader for the dataset with the specified batch size

    Parameters
    ----------
    batch_size : int
        Batch size of train loader
    dataset_name : str
         Name of the dataset (MNIST or FashionMNIST)

    Returns
    -------
    trainloader : DataLoader

    """
    transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])

    if dataset_name == "MNIST":
        trainset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
    elif dataset_name == "FashionMNIST":
        trainset = torchvision.datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return trainloader


class config_data:
    def __init__(self, dataset, image_size, channels, logit_transform, random_flip):
        self.dataset = dataset
        self.image_size = image_size
        self.channels = channels
        self.logit_transform = logit_transform
        self.random_flip = random_flip


class config_model:
    def __init__(self, num_classes, ngf):
        self.num_classes = num_classes
        self.ngf = ngf


class config:
    def __init__(
        self,
        dataset,
        image_size,
        channels,
        logit_transform,
        random_flip,
        num_classes,
        ngf,
    ):
        self.data = config_data(
            dataset, image_size, channels, logit_transform, random_flip
        )
        self.model = config_model(num_classes, ngf)
