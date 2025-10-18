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
from sklearn.metrics import roc_auc_score
import numpy as np


data_path = Path("./data")
data_path.mkdir(exist_ok=True)

model_path = Path("./models")
model_path.mkdir(exist_ok=True)


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


def train(
    model: nn.Module, trainloader: torch.utils.data.DataLoader, epochs=20, lr=0.001
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


def train_dirichlet():
    model = resnet18()
    model_save_name = "model_dirichlet.pth"
    trainloader, testloader = load_train_data()
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 10), nn.ReLU())

    model = train(model=model, trainloader=trainloader)
    model_save_path = Path.joinpath(model_path, model_save_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved the model to {model_save_path}.")


def get_uncertainties(models, device, dataloader, dataset_name):
    softmax_scores = {}
    all_targets = []

    for idx, model in enumerate(models):
        progress = tqdm.tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval {dataset_name} - Model {idx + 1}/{len(models)}",
            unit="batch",
        )
        softmax_scores[idx] = []
        for step, (inputs, target_label) in enumerate(progress, 1):
            inputs, target_label = inputs.to(device), target_label.to(device)
            with torch.no_grad():
                pred_labels = model(inputs)
            pred_label_scores = torch.softmax(pred_labels, -1)
            softmax_scores[idx].append(pred_label_scores)

            if idx == 0:
                all_targets.extend(target_label.cpu().tolist())

            torch.cuda.empty_cache()

        model.to("cpu")
        torch.cuda.empty_cache()

    stacked = [torch.cat(softmax_scores[idx], dim=0) for idx in sorted(softmax_scores)]
    scores = torch.stack(stacked, dim=0)
    scores = scores.transpose(0, 1)

    mean_pred = scores.mean(dim=1)
    ensemble_predictions = mean_pred.argmax(dim=1).cpu().numpy()

    uncertainties = compute_uncertainties(scores)
    all_targets = torch.tensor(all_targets).numpy()

    for model in models:
        model.to(device)

    return uncertainties, ensemble_predictions, all_targets


def compute_uncertainties(scores):
    mean_pred = scores.mean(dim=1)

    shan_entropies = -(scores * torch.log(scores + 1e-10)).sum(dim=-1)
    aleatoric = shan_entropies.mean(dim=1)

    mean_entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1)
    epistemic = mean_entropy - aleatoric

    return {
        "mean_pred": mean_pred,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "total": mean_entropy,
    }


def quantify_n_model_uncertainty():
    _, testloader = load_train_data()
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    models = load_all_models(device)

    uncertainties, ensemble_predictions, all_targets = get_uncertainties(
        models, device=device, dataloader=testloader, dataset_name="CIFAR10"
    )

    incorrect_predictions = (ensemble_predictions != all_targets).astype(int)

    for i in range(10):
        print(f"Sample {i}:")
        print(f"Predicted class: {uncertainties['mean_pred'][i].argmax()}")
        print(f"Aleatoric: {uncertainties['aleatoric'][i]:.4f}")
        print(f"Epistemic: {uncertainties['epistemic'][i]:.4f}")
        print(f"Total: {uncertainties['total'][i]:.4f}")

    total_uncertainty = uncertainties["total"]

    auc_score = roc_auc_score(incorrect_predictions, total_uncertainty.cpu().numpy())

    print(f"\nAUC Score (uncertainty vs misclassification): {auc_score:.4f}")


def load_all_models(device):
    models = []
    for single_model_path in Path.iterdir(model_path):
        model = resnet18(num_classes=10)
        model.load_state_dict(torch.load(single_model_path, weights_only=True))
        model.to(device)
        models.append(model)
    return models


def ood_detection():
    _, testloader_cifar10 = load_train_data()
    testloader_cifar10c = load_cifar10c(corruption_type="gaussian_noise", severity=5)
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    models = load_all_models(device)

    uncertainties_c10, ensemble_predictions_c10, all_targets_c10 = get_uncertainties(
        models, device=device, dataloader=testloader_cifar10, dataset_name="CIFAR10"
    )

    uncertainties_c10c, ensemble_predictions_c10c, all_targets_c10c = get_uncertainties(
        models, device=device, dataloader=testloader_cifar10c, dataset_name="CIFAR10c"
    )

    n_id = len(uncertainties_c10["epistemic"])
    n_ood = len(uncertainties_c10c["epistemic"])

    ood_labels = np.concatenate([np.zeros(n_id), np.ones(n_ood)])

    uncertainty_types = {
        "Total": torch.cat(
            [uncertainties_c10["total"], uncertainties_c10c["total"]], dim=0
        )
        .cpu()
        .numpy(),
        "Epistemic": torch.cat(
            [uncertainties_c10["epistemic"], uncertainties_c10c["epistemic"]], dim=0
        )
        .cpu()
        .numpy(),
        "Aleatoric": torch.cat(
            [uncertainties_c10["aleatoric"], uncertainties_c10c["aleatoric"]], dim=0
        )
        .cpu()
        .numpy(),
    }

    correct_predictions_c10 = np.mean(ensemble_predictions_c10 == all_targets_c10)
    correct_predictions_c10c = np.mean(ensemble_predictions_c10c == all_targets_c10c)

    print(f"Correct prediction percent for CIFAR10: {correct_predictions_c10}")
    print(f"Correct prediction percent for CIFAR10c: {correct_predictions_c10c}")

    print("OOD Detection Results:")
    print(f"ID samples: {n_id}, OOD samples: {n_ood}")

    for name, uncertainty in uncertainty_types.items():
        auroc = roc_auc_score(ood_labels, uncertainty)
        print(f"{name} Uncertainty AUROC: {auroc:.4f}")

        id_mean = uncertainty[:n_id].mean()
        ood_mean = uncertainty[n_id:].mean()
        print(f"  ID mean: {id_mean:.4f}, OOD mean: {ood_mean:.4f}")


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
    ood_detection()
