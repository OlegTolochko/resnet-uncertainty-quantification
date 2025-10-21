from data_manager import load_train_data
from train import train

from pathlib import Path

import torch
import torch.nn as nn
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
        Scalar loss value
    """
    one_hot_true_labels = torch.zeros_like(alpha)
    one_hot_true_labels.scatter_(1, true_labels, 1)
    alpha_0 = torch.sum(alpha, dim=-1, keepdim=True)
    pred_labels_distr = alpha/alpha_0 # convert to probabilities

    var = torch.sum(pred_labels_distr *(1- pred_labels_distr)/(alpha_0+1), dim=1, keepdim=True)
    err = torch.square((one_hot_true_labels-pred_labels_distr))
    L_base = torch.sum(err + var, dim=-1)

    concentration_mod = alpha
    concentration_mod.scatter_(1, true_labels, 1)
    uniform_distr = torch.ones_like(concentration_mod)
    dirichlet_pred = torch.distributions.Dirichlet(concentration_mod)
    dirichlet_uniform = torch.distributions.Dirichlet(uniform_distr)
    kl = torch.distributions.kl_divergence(dirichlet_pred, dirichlet_uniform)
    lambda_t = min(epoch/10, 1) # scaling factor to let model focus on fitting data in beginning

    loss = L_base + lambda_t*kl
    return torch.mean(loss)


@app.command()
def train_dirichlet():
    model = resnet18()
    model_save_name = "model_dirichlet.pth"
    trainloader, testloader = load_train_data()
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 10), nn.ReLU())

    model = train(model=model, trainloader=trainloader)
    model_save_path = Path.joinpath(dirichlet_model_path, model_save_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved the model to {model_save_path}.")


if __name__ == "__main__":
    app()
