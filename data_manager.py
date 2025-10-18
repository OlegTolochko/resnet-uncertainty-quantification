from pathlib import Path
from typing import List

import torch
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms
import numpy as np

data_path = Path("./data")
data_path.mkdir(exist_ok=True)


def load_cifar10c(corruption_type="jpeg_compression", severity=1, batch_size=128):
    cifar10c_path = Path("data/CIFAR-10-C")

    corruption_file = cifar10c_path / f"{corruption_type}.npy"
    labels_file = cifar10c_path / "labels.npy"

    images = np.load(corruption_file)
    labels = np.load(labels_file)

    start_idx = (severity - 1) * 10000
    end_idx = severity * 10000
    images = images[start_idx:end_idx]
    labels = labels[start_idx:end_idx]

    images = torch.from_numpy(images).float() / 255.0
    images = images.permute(0, 3, 1, 2)

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
    )
    images = torch.stack([normalize(img) for img in images])
    labels = torch.from_numpy(labels).long()

    dataset = torch.utils.data.TensorDataset(images, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    return dataloader


def load_train_data(
    dataset: torch.utils.data.Dataset = CIFAR10, batch_size: int = 128
) -> List[torch.utils.data.DataLoader]:
    """
    Args:
        dataset (Dataset): The dataset to load, Options: e.g. CIFAR10, CIFAR100
    """
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = dataset(
        root=data_path, train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=batch_size, shuffle=True
    )

    testset = dataset(
        root=data_path, train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=batch_size, shuffle=False
    )
    return trainloader, testloader
