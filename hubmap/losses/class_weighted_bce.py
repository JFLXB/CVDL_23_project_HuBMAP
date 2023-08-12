import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassWeightedBCELoss(nn.Module):
    def __init__(self, weights=None):
        super(ClassWeightedBCELoss, self).__init__()
        if weights is None:
            self._class_weights = None
        elif isinstance(weights, list):
            self._class_weights = torch.tensor(weights)
        elif isinstance(weights, torch.Tensor):
            self._class_weights = weights
        elif isinstance(weights, np.ndarray):
            self._class_weights = torch.tensor(weights)
        else:
            raise ValueError(
                "weights must be a list, torch.Tensor, or np.ndarray"
            )

    def forward(self, predictions, targets):
        n_classes = predictions.size(1)

        if self._class_weights is None:
            self._class_weights = torch.ones(n_classes)

        loss_per_class = torch.zeros(n_classes)
        for class_idx in range(n_classes):
            loss_per_class[class_idx] = F.binary_cross_entropy(
                predictions[:, class_idx], targets[:, class_idx]
            )

        loss = torch.dot(loss_per_class, self._class_weights)
        loss /= n_classes
        return loss


if __name__ == "__main__":
    should = nn.BCELoss()
    actual = ClassWeightedBCELoss()

    s = torch.rand((1, 4, 2, 2))
    a = torch.rand((1, 4, 2, 2))

    print("should: ", should(s, a))
    print("actual: ", actual(s, a))
    assert torch.isclose(should(s, a), actual(s, a))
