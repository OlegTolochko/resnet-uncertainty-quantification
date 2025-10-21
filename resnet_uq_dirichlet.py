from data_manager import load_train_data, load_cifar10c
from utilities import compute_uncertainties, ood_detection, BASE_MODEL_MAP, ModelType

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.models import resnet18
import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np
from cyclopts import App

app = App()

model_path = Path("./models")
model_path.mkdir(exist_ok=True)

dirichlet_model_path = Path.joinpath(model_path, "dirichlet")
dirichlet_model_path.mkdir(exist_ok=True)

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
    pred_labels_distr = alpha/alpha_0 # convert to probabilities

    var = torch.sum(pred_labels_distr *(1- pred_labels_distr)/(alpha_0+1), dim=1, keepdim=True)
    err = torch.square((one_hot_true_labels-pred_labels_distr))
    L_base = torch.sum(err + var, dim=-1)

    alpha_tilde = one_hot_true_labels + (1 - one_hot_true_labels) * alpha 
    uniform_distr = torch.ones_like(alpha_tilde)
    dirichlet_pred = torch.distributions.Dirichlet(alpha_tilde)
    dirichlet_uniform = torch.distributions.Dirichlet(uniform_distr)
    kl = torch.distributions.kl_divergence(dirichlet_pred, dirichlet_uniform)
    lambda_t = min(epoch/10, 1) # scaling factor to let model focus on fitting data in beginning

    loss = L_base + lambda_t*kl
    return torch.mean(loss)


def train_dirichlet_model(
    model: nn.Module, trainloader: torch.utils.data.DataLoader, epochs=50, lr=0.001
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
def train_dirichlet():
    model = resnet18()
    model_save_name = "model_dirichlet.pth"
    trainloader, testloader = load_train_data()
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 10), nn.ReLU())

    model = train_dirichlet_model(model=model, trainloader=trainloader)
    model_save_path = Path.joinpath(dirichlet_model_path, model_save_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved the model to {model_save_path}.")


if __name__ == "__main__":
    app()
