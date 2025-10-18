import torch


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
