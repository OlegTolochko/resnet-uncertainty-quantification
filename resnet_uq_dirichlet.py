from data_manager import load_train_data
from train import train

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import resnet18
import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

model_path = Path("./models")
model_path.mkdir(exist_ok=True)

dirichlet_model_path = Path.joinpath(model_path, "dirichlet")
dirichlet_model_path.mkdir(exist_ok=True)


def train_dirichlet():
    model = resnet18()
    model_save_name = "model_dirichlet.pth"
    trainloader, testloader = load_train_data()
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 10), nn.ReLU())

    model = train(model=model, trainloader=trainloader)
    model_save_path = Path.joinpath(dirichlet_model_path, model_save_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved the model to {model_save_path}.")
