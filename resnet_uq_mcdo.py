from train import train
from data_manager import load_train_data, load_cifar10c
from utilities import compute_uncertainties, ood_detection, BASE_MODEL_MAP, ModelType

from pathlib import Path

import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import roc_auc_score
from cyclopts import App

app = App()

MCDO_MODEL_PATH = Path("./models/mcdo")
MCDO_MODEL_PATH.mkdir(parents=True, exist_ok=True)


class ResNetMCDO(nn.Module):
    """ResNet with MC Dropout for uncertainty quantification."""

    def __init__(self, base_model, num_classes=10, dropout_p=0.3):
        super().__init__()
        self.resnet = base_model()
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_p), nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


def create_mcdo_model(
    model_type: str, num_classes: int = 10, dropout_p: float = 0.3
) -> ResNetMCDO:
    """Factory for ResNetMCDO"""
    if model_type not in BASE_MODEL_MAP:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported: {list(BASE_MODEL_MAP.keys())}"
        )
    base_model_fn = BASE_MODEL_MAP[model_type]
    return ResNetMCDO(base_model_fn, num_classes=num_classes, dropout_p=dropout_p)


def load_mcdo_model(
    model_type: str,
    model_name: str,
    device: torch.device,
    num_classes: int = 10,
    dropout_p: float = 0.3,
) -> ResNetMCDO:
    model = create_mcdo_model(model_type, num_classes=num_classes, dropout_p=dropout_p)
    model_path = MCDO_MODEL_PATH / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.to(device)
    model.train()  # Keeps dropout active
    return model


@app.command()
def train_mcdo(
    model_type: ModelType = "resnet18",
    dropout_p: float = 0.3,
    save_name: str = None,
    n_epochs: int = 100,
):
    """Train a ResNet with MC Dropout."""
    model = create_mcdo_model(model_type, dropout_p=dropout_p)

    if not save_name:
        save_name = f"{model_type}_mcdo_ep{n_epochs}_{int(dropout_p * 100)}dp.pth"

    trainloader, _ = load_train_data()
    model = train(model=model, trainloader=trainloader, epochs=n_epochs)
    torch.save(model.state_dict(), MCDO_MODEL_PATH / save_name)
    print(f"Saved model to {MCDO_MODEL_PATH / save_name}")


def get_uncertainties_mcdo(model, n_iter, device, dataloader, dataset_name):
    """Run MC Dropout inference and compute uncertainties"""
    softmax_scores = {}
    all_targets = []

    for idx in range(n_iter):
        progress = tqdm.tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval {dataset_name} - Model {idx + 1}/{n_iter}",
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

        torch.cuda.empty_cache()

    stacked = [torch.cat(softmax_scores[idx], dim=0) for idx in sorted(softmax_scores)]
    scores = torch.stack(stacked, dim=0)
    scores = scores.transpose(0, 1)

    mean_pred = scores.mean(dim=1)
    ensemble_predictions = mean_pred.argmax(dim=1).cpu().numpy()

    uncertainties = compute_uncertainties(scores)
    all_targets = torch.tensor(all_targets).numpy()

    return uncertainties, ensemble_predictions, all_targets


@app.command()
def quantify_mcdo_model_uncertainty(
    model_type: ModelType = "resnet18",
    model_name: str = "resnet18_mcdo_ep100_30dp.pth",
    dropout_p: float = 0.3,
):
    """Quantify uncertainty and compute AUC for misclassification detection."""
    _, testloader = load_train_data()
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = load_mcdo_model(model_type, model_name, device, dropout_p=dropout_p)

    uncertainties, predictions, targets = get_uncertainties_mcdo(
        model, n_iter=10, device=device, dataloader=testloader, dataset_name="CIFAR-10"
    )

    for i in range(10):
        print(f"\nSample {i}:")
        print(f"  Predicted: {predictions[i]}, True: {targets[i]}")
        print(f"  Aleatoric: {uncertainties['aleatoric'][i]:.4f}")
        print(f"  Epistemic: {uncertainties['epistemic'][i]:.4f}")
        print(f"  Total: {uncertainties['total'][i]:.4f}")

    misclassified = (predictions != targets).astype(int)
    auc = roc_auc_score(misclassified, uncertainties["total"].numpy())
    print(f"\nAUC (uncertainty vs misclassification): {auc:.4f}")


@app.command()
def ood_detection_mcdo(
    model_type: ModelType = "resnet18",
    model_name: str = "resnet_mcdo_ep100_30dp.pth",
    dropout_p: float = 0.3,
    corruption_type: str = "gaussian_noise",
    corruption_severity: int = 2,
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
    model = load_mcdo_model(model_type, model_name, device, dropout_p=dropout_p)

    unc_c10, pred_c10, target_c10 = get_uncertainties_mcdo(
        model,
        n_iter=10,
        device=device,
        dataloader=testloader_c10,
        dataset_name="CIFAR-10",
    )
    unc_c10c, pred_c10c, target_c10c = get_uncertainties_mcdo(
        model,
        n_iter=10,
        device=device,
        dataloader=testloader_c10c,
        dataset_name="CIFAR-10-C",
    )

    ood_detection(unc_c10, unc_c10c, pred_c10, pred_c10c, target_c10, target_c10c)


if __name__ == "__main__":
    app()
