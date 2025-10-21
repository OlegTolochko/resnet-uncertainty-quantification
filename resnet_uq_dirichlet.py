from data_manager import load_train_data, load_cifar10c
from utilities import compute_uncertainties, ood_detection, BASE_MODEL_MAP, ModelType

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.special import digamma
from torchvision.models import resnet18
import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
from cyclopts import App

app = App()

model_path = Path("./models")
model_path.mkdir(exist_ok=True)

DIRICHLET_MODEL_PATH = Path.joinpath(model_path, "dirichlet")
DIRICHLET_MODEL_PATH.mkdir(exist_ok=True)


class ResNetDirichlet(nn.Module):
    """ResNet with MC Dropout for uncertainty quantification."""

    def __init__(self, base_model, num_classes=10):
        super().__init__()
        self.resnet = base_model()
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(nn.Linear(num_features, num_classes), nn.ReLU())

    def forward(self, x):
        return self.resnet(x)


def create_dirichlet_model(model_type: str, num_classes: int = 10) -> ResNetDirichlet:
    """Factory for ResNetMCDO"""
    if model_type not in BASE_MODEL_MAP:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported: {list(BASE_MODEL_MAP.keys())}"
        )
    base_model_fn = BASE_MODEL_MAP[model_type]
    return ResNetDirichlet(base_model_fn, num_classes=num_classes)


def dirichlet_loss(alpha, true_labels, epoch):
    """
    Custom loss to punish uncertainty in predictions, as well as overconfident incorrect predictions.
    The Loss enforces the model to output meaningful dirichlet concentration parameters for the dirichlet distribution.

    Args:
        alpha: Dirichlet concentration parameters, shape [batch_size, n_classes]
        true_labels: Ground truth class labels, shape [batch_size]
        epoch: Current training epoch

    Returns:
        Loss Value
    """
    n_classes = alpha.shape[-1]
    one_hot_true_labels = F.one_hot(true_labels, num_classes=n_classes)
    alpha_0 = torch.sum(alpha, dim=-1, keepdim=True)
    pred_labels_distr = alpha / alpha_0  # convert to probabilities

    var = torch.sum(
        pred_labels_distr * (1 - pred_labels_distr) / (alpha_0 + 1), dim=1, keepdim=True
    )
    err = torch.square((one_hot_true_labels - pred_labels_distr))
    L_base = torch.sum(err + var, dim=-1)

    alpha_tilde = one_hot_true_labels + (1 - one_hot_true_labels) * alpha
    uniform_distr = torch.ones_like(alpha_tilde)
    dirichlet_pred = torch.distributions.Dirichlet(alpha_tilde)
    dirichlet_uniform = torch.distributions.Dirichlet(uniform_distr)
    kl = torch.distributions.kl_divergence(dirichlet_pred, dirichlet_uniform)
    lambda_t = min(
        epoch / 10, 1
    )  # scaling factor to let model focus on fitting data in beginning

    loss = L_base + lambda_t * kl
    return torch.mean(loss)


def train_dirichlet_model(
    model: ResNetDirichlet,
    trainloader: torch.utils.data.DataLoader,
    epochs=100,
    lr=0.001,
):
    """
    Adjusted training setup for dirichlet distribution outputting models.

    Args:
        model (Model): The model to load, Options: e.g. resnet18, resnet50, VGG19
    """
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    criterion = dirichlet_loss
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

            evidence = model(inputs)
            concentration = evidence + 1
            loss = criterion(concentration, target_label, epoch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = concentration.max(1)
            total += target_label.size(0)
            correct += predicted.eq(target_label).sum().item()

            progress.set_postfix(
                loss=train_loss / step,
                acc=100.0 * correct / total,
                correct=correct,
                total=total,
            )

    return model


@app.command()
def train_dirichlet(
    model_type: ModelType = "resnet18", save_name: str = None, n_epochs: int = 100
):
    """Train a ResNet with MC Dropout."""
    model = create_dirichlet_model(model_type)

    if not save_name:
        save_name = f"{model_type}_dirichlet_ep{n_epochs}.pth"

    trainloader, _ = load_train_data()
    model = train_dirichlet_model(model=model, trainloader=trainloader, epochs=n_epochs)
    torch.save(model.state_dict(), DIRICHLET_MODEL_PATH / save_name)
    print(f"Saved model to {DIRICHLET_MODEL_PATH / save_name}")


def load_dirichlet_model(
    model_type: str, model_name: str, device: torch.device, num_classes: int = 10
) -> ResNetDirichlet:
    model = create_dirichlet_model(model_type, num_classes=num_classes)
    model_path = DIRICHLET_MODEL_PATH / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    return model


def compute_dirichlet_uncertainties(alpha):
    alpha0 = torch.sum(alpha, dim=-1, keepdim=True)
    mean_pred = alpha.mean(dim=1)

    scores = alpha / alpha0
    total_uncertainty = -(scores * torch.log(scores + 1e-10)).sum(dim=-1)

    aleatoric_uncertainty = digamma(alpha0 + 1).squeeze(-1) - (
        1 / alpha0.squeeze(-1)
    ) * (alpha * digamma(alpha + 1)).sum(dim=-1)
    epistemic = total_uncertainty - aleatoric_uncertainty
    return {
        "mean_pred": mean_pred,
        "aleatoric": aleatoric_uncertainty,
        "epistemic": epistemic,
        "total": total_uncertainty,
    }


def get_uncertainties_dirichlet(model, device, dataloader, dataset_name):
    """Run MC Dropout inference and compute uncertainties"""
    dirichlet_concentrations = []
    all_targets = []
    progress = tqdm.tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"Eval {dataset_name}",
        unit="batch",
    )
    for step, (inputs, target_label) in enumerate(progress, 1):
        inputs, target_label = inputs.to(device), target_label.to(device)
        with torch.no_grad():
            evidence = model(inputs)
        dirichlet_concentrations.append(evidence + 1)

        all_targets.extend(target_label.cpu().tolist())

        torch.cuda.empty_cache()

    alphas = torch.cat(dirichlet_concentrations)

    ensemble_predictions = alphas.argmax(dim=1).cpu().numpy()

    uncertainties = compute_dirichlet_uncertainties(alphas)

    return uncertainties, ensemble_predictions, all_targets


@app.command()
def quantify_dirichlet_model_uncertainty(
    model_type: ModelType = "resnet18",
    model_name: str = "resnet18_dirichlet_ep100.pth",
):
    """Quantify uncertainty and compute AUC for misclassification detection."""
    _, testloader = load_train_data()
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_dirichlet_model(model_type, model_name, device)

    uncertainties, predictions, targets = get_uncertainties_dirichlet(
        model, device=device, dataloader=testloader, dataset_name="CIFAR-10"
    )

    for i in range(10):
        print(f"\nSample {i}:")
        print(f"  Predicted: {predictions[i]}, True: {targets[i]}")
        print(f"  Aleatoric: {uncertainties['aleatoric'][i]:.4f}")
        print(f"  Epistemic: {uncertainties['epistemic'][i]:.4f}")
        print(f"  Total: {uncertainties['total'][i]:.4f}")

    misclassified = (predictions != targets).astype(int)
    auc = roc_auc_score(misclassified, uncertainties["total"].cpu().numpy())
    print(f"\nAUC (uncertainty vs misclassification): {auc:.4f}")


@app.command()
def ood_detection_dirichlet(
    model_type: ModelType = "resnet18",
    model_name: str = "resnet18_dirichlet_ep100.pth",
    corruption_type: str = "fog",
    corruption_severity: int = 5,
):
    """Perform OOD detection using CIFAR-10 vs CIFAR-10-C"""
    _, testloader_c10 = load_train_data()
    testloader_c10c = load_cifar10c(
        corruption_type=corruption_type, severity=corruption_severity
    )
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_dirichlet_model(model_type, model_name, device)

    unc_c10, pred_c10, target_c10 = get_uncertainties_dirichlet(
        model, device=device, dataloader=testloader_c10, dataset_name="CIFAR-10"
    )
    unc_c10c, pred_c10c, target_c10c = get_uncertainties_dirichlet(
        model, device=device, dataloader=testloader_c10c, dataset_name="CIFAR-10-C"
    )

    ood_detection(unc_c10, unc_c10c, pred_c10, pred_c10c, target_c10, target_c10c)


if __name__ == "__main__":
    app()
