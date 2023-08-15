"""
IoU Metric
"""
import torch
from copy import deepcopy


class IoU:
    """ """

    @property
    def name(self):
        return self._name

    def __init__(
        self,
        name: str = "IoU",
        class_index: int = None,
        reduction: str = "mean",
        pred_idx: int = None,
        activation_fun=None,
    ):
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Batch Reduction {reduction} is not supported. Please choose one of ['mean', 'sum', 'none']"
            )
        self._classes_to_evaluate = class_index
        self._reduction = reduction
        self._pred_idx = pred_idx
        self._activation_fun = activation_fun
        self._name = name

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
        prediction = prediction.clone().detach().cpu()
        target = target.clone().detach().cpu()
        prediction = (
            prediction[self._pred_idx] if self._pred_idx is not None else prediction
        )

        if target.size() != prediction.size():
            raise ValueError(
                f"Target and prediction must have the same size, but got {target.size()} and {prediction.size()}."
            )
        if target.dim() != 4:
            target = target.unsqueeze(0)
        if prediction.dim() != 4:
            prediction = prediction.unsqueeze(0)

        if self._activation_fun:
            prediction = self._activation_fun(prediction)

        if self._classes_to_evaluate is None:
            n_classes = target.size(1)
            iou_score = torch.zeros((target.size(0), n_classes))
            for i in range(n_classes):
                iou_class_scores = self._calculate_iou_for_class(i, target, prediction)
                iou_score[:, i] = iou_class_scores.view(-1)
            iou_score = self._do_reduction(iou_score, dim=1)
        else:
            iou_score = self._calculate_iou_for_class(
                self._classes_to_evaluate, target, prediction
            )

        return self._do_reduction(iou_score, dim=0)

    def _calculate_iou_for_class(self, cls_idx: int, T: torch.Tensor, P: torch.Tensor):
        T_cls = T[:, cls_idx : cls_idx + 1, :, :].clone()
        P_cls = P[:, cls_idx : cls_idx + 1, :, :].clone()
        return self._calculate_iou_over_batches(T_cls, P_cls)

    def _calculate_iou_over_batches(self, T: torch.Tensor, P: torch.Tensor):
        T, P = T.clone(), P.clone()
        is_special_case = T.sum(dim=(-2, -1)) == 0
        T[is_special_case] = torch.logical_not(T[is_special_case]).type(torch.float32)
        P[is_special_case] = torch.logical_not(P[is_special_case]).type(torch.float32)
        intersection = torch.logical_and(T, P).sum(dim=(-2, -1))
        uninon = torch.logical_or(T, P).sum(dim=(-2, -1))
        iou = intersection / uninon
        return iou

    def _do_reduction(self, iou_scores, dim):
        if self._reduction == "mean":
            return torch.mean(iou_scores, dim=dim)
        elif self._reduction == "sum":
            return torch.sum(iou_scores, dim=dim)
        else:
            return iou_scores


if __name__ == "__main__":
    target = torch.tensor(
        [
            [[[0, 0], [0, 0]], [[1, 0], [1, 0]]],
            [
                [[0, 1], [0, 0]],
                [[0, 0], [1, 1]],
            ],
            [
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
        ],
        dtype=torch.float32,
    )

    pred = torch.tensor(
        [
            [[[0, 0], [0, 0]], [[0, 0], [1, 0]]],
            [
                [[1, 1], [1, 0]],
                [[0, 0], [1, 0]],
            ],
            [
                [[0, 0], [0, 0]],
                [[0, 0], [0, 0]],
            ],
        ],
        dtype=torch.float32,
    )

    iou = IoU()

    score = iou(target, pred)

    goal = torch.tensor([[1.0000, 0.5000], [0.3333, 0.5000], [1.0000, 1.0000]])
    assert torch.isclose(score, goal.mean(1).mean(0))

    from sklearn.metrics import jaccard_score

    jac = jaccard_score(target.view(-1), pred.view(-1), average="weighted")
    print(jac)
    print(score)
