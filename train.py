import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


def train(
    model: nn.Module, trainloader: torch.utils.data.DataLoader, epochs=50, lr=0.001
):
    """
    Basic training setup for any type of model.

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
