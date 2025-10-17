from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.special import digamma
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


def train_dirichlet():
    model = resnet18()
    model_save_name = "model_dirichlet.pth"
    trainloader, testloader = load_train_data() 
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 10), nn.ReLU())

    model = train(model=model, trainloader=trainloader)
    model_save_path = Path.joinpath(model_path, model_save_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved the model to {model_save_path}.")


def quantify_n_model_uncertainty():
    _, testloader = load_train_data()
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    models = []
    for single_model_path in Path.iterdir(model_path):
        model = resnet18(num_classes=10)
        model.load_state_dict(torch.load(single_model_path, weights_only=True))
        model.to(device)
        models.append(model)

    total = 0
    correct = 0
    softmax_scores = {}
    for idx, model in enumerate(models):
        progress = tqdm.tqdm(
            testloader,
            total=len(testloader),
            desc=f"Eval Model: {idx + 1}/{len(models)}",
            unit="batch",
        )
        softmax_scores[idx] = []
        for step, (inputs, target_label) in enumerate(progress, 1):
            inputs, target_label = inputs.to(device), target_label.to(device)
            with torch.no_grad():
                pred_labels = model(inputs)
            pred_label_scores = torch.softmax(pred_labels, -1)
            softmax_scores[idx].append(pred_label_scores)
            _, predicted = pred_labels.max(1)
            total += target_label.size(0)
            correct += predicted.eq(target_label).sum().item()

            progress.set_postfix(
                acc=100.0 * correct / total,
                correct=correct,
                total=total,
            )
            torch.cuda.empty_cache()

        model.to("cpu")
        del model

    stacked = [torch.cat(softmax_scores[idx], dim=0) for idx in sorted(softmax_scores)]
    scores = torch.stack(stacked, dim=0)
    scores = scores.transpose(0, 1) # [Batch, n_models, n_classes]

    uncertainties = compute_uncertainties(scores)

    for i in range(10):
        print(f"Sample {i}:")
        print(f"  Predicted class: {uncertainties['mean_pred'][i].argmax()}")
        print(f"  Aleatoric: {uncertainties['aleatoric'][i]:.4f}")
        print(f"  Epistemic: {uncertainties['epistemic'][i]:.4f}")
        print(f"  Total: {uncertainties['total'][i]:.4f}")
 
    concentration_params = get_dirichlet_concentration_params(prob_vectors=scores)
    distribution = torch.distributions.dirichlet.Dirichlet(concentration_params[0])
    print(f"Mean: {distribution.mean}")
    print(f"Variance: {distribution.entropy()}")

    alpha = concentration_params[0]
    alpha0 = alpha.sum(dim=-1, keepdim=True)

    mean_p = alpha / alpha0
    pred_entropy = -(mean_p * torch.log(mean_p + 1e-12)).sum(dim=-1)

    aleatoric_unc = (
        digamma(alpha0 + 1)
        - (1 / alpha0.squeeze(-1)) * (alpha * digamma(alpha + 1)).sum(dim=-1)
    )

    epistemic = pred_entropy - aleatoric_unc
    print(f"epistemic: {epistemic}")
    print(f"pred entropy: {pred_entropy}")
    print(f"alaetoric: {aleatoric_unc}")

def compute_uncertainties(scores):
    mean_pred = scores.mean(dim=1)
    
    individual_entropies = -(scores * torch.log(scores + 1e-10)).sum(dim=-1)
    aleatoric = individual_entropies.mean(dim=1)
    
    mean_entropy = -(mean_pred * torch.log(mean_pred + 1e-10)).sum(dim=-1)
    epistemic = mean_entropy - aleatoric
    
    return {
        'mean_pred': mean_pred,
        'aleatoric': aleatoric,
        'epistemic': epistemic,
        'total': mean_entropy
    }




def get_dirichlet_concentration_params(prob_vectors: torch.Tensor):
    """
    Args:
        prob_vectors (Probability Vectors): Need to be of shape (Batch, Num Models, Num Classes)
    """
    sample_means = torch.mean(prob_vectors, dim=1)
    sample_vars = torch.mean(prob_vectors, dim=1)

    alpha0s = (sample_means * (1 - sample_means)) / (sample_vars)
    alpha0_means = torch.mean(alpha0s, dim=-1)

    concentration_params = sample_means * alpha0_means[:, None]
    return concentration_params


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
    quantify_n_model_uncertainty()
