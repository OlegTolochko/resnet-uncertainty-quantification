from typing import Literal, Dict, Callable

import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision.models import resnet18, resnet50, resnet34, wide_resnet50_2


BASE_MODEL_MAP: Dict[str, Callable] = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "wide_resnet50": wide_resnet50_2
    # More model architectures may be added here
}
ModelType = Literal["resnet18", "resnet34", "resnet50", "wide_resnet50"]


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


def ood_detection(
    uncertainties_c10,
    uncertainties_c10c,
    ensemble_predictions_c10,
    ensemble_predictions_c10c,
    all_targets_c10,
    all_targets_c10c,
):
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
