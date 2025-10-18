from train import train
from data_manager import load_train_data, load_cifar10c
from uq_helper import compute_uncertainties

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18
import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


model_path = Path("./models")
model_path.mkdir(exist_ok=True)

mcdo_model_path = Path.joinpath(model_path, "mcdo")
mcdo_model_path.mkdir(exist_ok=True)


class ResNet18_MCDO(nn.Module):
    def __init__(self, num_classes=10, dropout_p=0.3):
        super(ResNet18_MCDO, self).__init__()
        self.resnet = resnet18(pretrained=False)

        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=dropout_p), nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)


def train_mcdo(dropout_p=0.3):
    model = ResNet18_MCDO(dropout_p=dropout_p)
    model_save_name = f"model_mcdo_{int(dropout_p * 100)}dp.pth"
    trainloader, testloader = load_train_data()

    model = train(model=model, trainloader=trainloader)
    model_save_path = Path.joinpath(mcdo_model_path, model_save_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved the model to {model_save_path}.")


def get_uncertainties_mcdo(model, n_iter, device, dataloader, dataset_name):
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


def quantify_mcdo_model_uncertainty(model_name: str = "model_mcdo_30dp.pth"):
    _, testloader = load_train_data()
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = resnet18()
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 10))
    model.load_state_dict(
        torch.load(Path.joinpath(mcdo_model_path, model_name), weights_only=True)
    )
    model.to(device)
    model.train()

    uncertainties, ensemble_predictions, all_targets = get_uncertainties_mcdo(
        model, n_iter=10, device=device, dataloader=testloader, dataset_name="CIFAR10"
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
