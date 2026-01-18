import torch
import torch.nn as nn

def crossentropy_loss(predictions, targets, ignore_index=None):
    """
    Computes the cross-entropy loss between predictions and targets.

    Args:
        predictions (torch.Tensor): The predicted logits with shape (batch_size, num_classes).
        targets (torch.Tensor): The ground truth labels with shape (batch_size,).
        ignore_index (int, optional): Specifies a target value that is ignored

    """
    criterion = nn.CrossEntropyLoss()
    loss_ce = criterion(logits, targets)
    print(f"Cross-Entropy Loss: {loss_ce.item()}")
    print(f"Cross-Entropy Loss: {loss_ce}")

if __name__ == "__main__":
    logits = torch.tensor([
        [0.2, 0.8, 0.1],
        [0.5, 0.4, 0.1],
        [0.9, 0.05, 0.05]
    ])
    targets = torch.tensor([1, 1, 2])

    crossentropy_loss(logits, targets)