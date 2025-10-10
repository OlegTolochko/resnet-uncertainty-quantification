from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import resnet18, resnet50, vgg19


data_path = Path("./data")
data_path.mkdir(exist_ok=True)

def load_train_data(dataset: torch.utils.data.Dataset = CIFAR10):
    """
    Args:
        dataset (Dataset): The dataset to load, Options: e.g. CIFAR10, CIFAR100
    """
    trainset = dataset(root=data_path, train=True, download=True)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True)

    testset = dataset(root=data_path, train=False, download=True)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, shuffle=False)
    return trainloader, testloader

def train(model: nn.Module = resnet18, lr=0.001):
    """
    Args:
        model (Model): The model to load, Options: e.g. resnet18, resnet50, VGG19
    """
    trainloader, testloader = load_train_data()
    optimizer = optim.AdamW(lr=0.001)

    for batch in trainloader:
        pass