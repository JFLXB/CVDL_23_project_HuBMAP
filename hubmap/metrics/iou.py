"""
IoU Metric
"""
import torch
import torch.nn.functional as F
import numpy as np


class IoU:
    """ """

    def __init__(
        self,
        class_index: int = None,
        reduction: str = "mean",
    ):
        self._classes_to_evaluate = class_index
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Reduction {reduction} is not supported. Please choose one of ['mean', 'sum', 'none']"
            )
        self._reduction = reduction

    def __call__(self, prediction: torch.Tensor, target: torch.Tensor):
        """_summary_

        Parameters
        ----------
        prediction : _type_
            _description_
        target : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # prediction = prediction[2]
        probs = F.sigmoid(prediction.type(torch.float32))
        classes = torch.argmax(probs, dim=1, keepdims=True)
        classes_per_channel = torch.zeros_like(prediction)
        classes_per_channel.scatter_(1, classes, 1)
        
        if self._classes_to_evaluate:
            prediction_bv = classes_per_channel[:, self._classes_to_evaluate, :, :]
            target_bv = target[:, self._classes_to_evaluate, :, :]
            intersection = torch.logical_and(prediction_bv, target_bv)
            union = torch.logical_or(prediction_bv, target_bv)
            if torch.sum(union).item() == 0:
                iou_score = torch.tensor(0.0)
            else:
                iou_score = torch.sum(intersection, dim=(1, 2)) / torch.sum(union, dim=(1, 2))
        else:
            scores = torch.zeros(classes_per_channel.size(1))
            for i in range(classes_per_channel.size(1)):
                prediction_bv = classes_per_channel[:, i, :, :]
                target_bv = target[:, i, :, :]
                intersection = torch.logical_and(prediction_bv, target_bv)
                union = torch.logical_or(prediction_bv, target_bv)
                if torch.sum(union).item() == 0:
                    iou_score = torch.tensor(0.0)
                else:
                    iou_score = torch.sum(intersection, dim=(1, 2)) / torch.sum(union, dim=(1, 2))
                scores[i] = iou_score.item()
            iou_score = torch.mean(scores)

        # print(iou_score.size())
        # print(iou_score)

        # iou_score = torch.nan_to_num(iou_score, nan=0.0, posinf=0.0, neginf=0.0)

        # intersection = torch.logical_and(prediction, target)
        # union = torch.logical_or(prediction, target)
        # iou_score = torch.sum(intersection) / torch.sum(union)
        # # return self._do_reduction(iou_score)
        # print(iou_score.item())
        # return iou_score

        return self._do_reduction(iou_score)

    def _do_reduction(self, iou_scores):
        if self._reduction == "mean":
            return torch.mean(iou_scores)
        elif self._reduction == "sum":
            return torch.sum(iou_scores)
        else:
            return iou_scores


if __name__ == "__main__":
    iou = IoU(0)

    GT = torch.tensor([[[[1, 1, 0], [1, 1, 0], [0, 0, 0]]]])
    A = torch.tensor([[[[1, 1, 0], [1, 1, 0], [0, 0, 0]]]])
    assert iou(GT, A).item() == 1.0

    B = torch.tensor([[[[0, 1, 0], [0, 0, 0], [0, 0, 0]]]])
    assert iou(GT, B).item() == 1 / 4

    C = torch.tensor([[[[0, 1, 0], [0, 1, 0], [0, 0, 0]]]])
    assert iou(GT, C).item() == 2 / 4

    D = torch.tensor([[[[0, 1, 0], [1, 1, 0], [0, 0, 0]]]])
    assert iou(GT, D).item() == 3 / 4
