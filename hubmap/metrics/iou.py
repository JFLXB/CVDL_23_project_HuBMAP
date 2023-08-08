"""
IoU Metric
"""
import torch


class IoU:
    """ """

    def __init__(
        self,
        class_index: int,
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
        prediction_bv = prediction[:, self._classes_to_evaluate, :, :]
        target_bv = target[:, self._classes_to_evaluate, :, :]
        intersection = torch.logical_and(prediction_bv, target_bv)
        union = torch.logical_or(prediction_bv, target_bv)
        iou_score = torch.sum(intersection, dim=(1, 2)) / torch.sum(union, dim=(1, 2))
        return self._do_reduction(iou_score)

    def _do_reduction(self, iou_scores):
        if self._reduction == "mean":
            return torch.mean(iou_scores)
        elif self._reduction == "sum":
            return torch.sum(iou_scores)
        else:
            return iou_scores
