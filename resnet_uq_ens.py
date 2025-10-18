from data_manager import load_train_data, load_cifar10c
from train import train
from uq_helper import compute_uncertainties

from pathlib import Path

import torch
from torchvision.models import resnet18, resnet50, vgg19
import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np

model_path = Path("./models")
model_path.mkdir(exist_ok=True)


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


def quantify_n_model_uncertainty():
    _, testloader = load_train_data()
    device = (
        "mps"
        if torch.backends.mps.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    models = load_all_models_ens(device)

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


def load_all_models_ens(device):
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
    models = load_all_models_ens(device)

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


if __name__ == "__main__":
    quantify_n_model_uncertainty()
