from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.models import resnet18, resnet50, vgg19
from torchvision.transforms import transforms
import tqdm


data_path = Path("./data")
data_path.mkdir(exist_ok=True)

model_path = Path("./models")
model_path.mkdir(exist_ok=True)


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


def train(
    model: nn.Module, trainloader: torch.utils.data.DataLoader, epochs=50, lr=0.001
):
    """
    Args:
        model (Model): The model to load, Options: e.g. resnet18, resnet50, VGG19
    """
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    optimizer = optim.AdamW(params=model.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loss = 0.0
        correct = 0
        total = 0

        progress = tqdm.tqdm(
            trainloader,
            total=len(trainloader),
            desc=f"Train {epoch + 1}/{epochs}",
            unit="batch",
        )
        for step, (inputs, target_label) in enumerate(progress, 1):
            inputs, target_label = inputs.to(device), target_label.to(device)
            optimizer.zero_grad()

            pred_labels = model(inputs)
            loss = criterion(pred_labels, target_label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = pred_labels.max(1)
            total += target_label.size(0)
            correct += predicted.eq(target_label).sum().item()

            progress.set_postfix(
                loss=train_loss / step,
                acc=100.0 * correct / total,
                correct=correct,
                total=total,
            )

    return model


def train_dirichlet(
    model: nn.Module, trainloader: torch.utils.data.DataLoader, lr=0.001
):
    pass


def quantify_n_model_uncertainty():
    _, testloader = load_train_data()
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    models = []

    for idx, model in enumerate(models):
        progress = tqdm.tqdm(
            testloader,
            total=len(testloader),
            desc=f"Eval Model{idx + 1}/{len(models)}",
            unit="batch",
        )
        for step, (inputs, target_label) in enumerate(progress, 1):
            inputs, target_label = inputs.to(device), target_label.to(device)

            pred_labels = model(inputs)
            torch.softmax(pred_labels)
            _, predicted = pred_labels.max(1)
            total += target_label.size(0)
            correct += predicted.eq(target_label).sum().item()

            progress.set_postfix(
                acc=100.0 * correct / total,
                correct=correct,
                total=total,
            )


def train_n_models(n_models: int = 10, model_name: str = "resnet_model"):
    for i in range(n_models):
        model_save_name = model_name
        trainloader, testloader = load_train_data()
        model = resnet18(num_classes=10)
        train(model=model, trainloader=trainloader)
        model_save_name += f"_{i + 1}"
        model_save_name += ".pth"
        model_save_path = Path.joinpath(model_path, model_save_name)

        torch.save(model.state_dict(), model_save_path)
        print(f"Saved the model to {model_save_path}.")


def main():
    trainloader, testloader = load_train_data()
    model = resnet18(num_classes=10)
    train(model=model, trainloader=trainloader)


if __name__ == "__main__":
    train_n_models()
