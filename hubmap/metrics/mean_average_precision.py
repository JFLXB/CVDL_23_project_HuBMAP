"""
Submissions are evaluated by computing the Average Precision over confidence scores. 
It is identical to the OpenImages Instance Segmentation Challenge evaluation [1], 
though with only a single class. The OpenImages version of the metric is described in 
detail here [2]. See also this tutorial [3] on running the evaluation in Python.

Segmentation is calculated using IoU with a threshold of 0.6.

References
[1] https://www.kaggle.com/c/open-images-2019-instance-segmentation/overview/evaluation
[2] https://storage.googleapis.com/openimages/web/evaluation.html#instance_segmentation_eval
[3] https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/challenge_evaluation.md#instance-segmentation-track
"""
import torch
import torch.nn.functional as F


class MeanAveragePrecision:
    """
    Formula: mAP = 1/n * sum(AP_i)
    where n is the number of classes and AP_i is the Average Precision for class i.
    
    We only care about the Mean Average Precision (mAP) for the `blood_vessel` class.
    
    Sources:
        https://www.v7labs.com/blog/mean-average-precision#h2
        https://hasty.ai/docs/mp-wiki/metrics/map-mean-average-precision#mean-average-precision-explained
        
    """
    def __init__(self, blood_vessel_channel: int, iou_threshold: float = 0.6) -> None:
        self._blood_vessel_channel = blood_vessel_channel
        self._iou_threshold = iou_threshold
    
    def calculate(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Implements the mean average precision calculation.
    
        Steps:
            1. Generate the prediction scores using the model (done outside of this function)
            2. Determine the per-pixel class labels
            3. Calculate the confusion matrix
            4. Calculate the precision and recall metrics
            5. Calculate the area under the precision-recall curve
            6. Measure the average precision
        """
        prediction_labels = self._determine_per_pixel_label(prediction)
        target_labels = self._determine_per_pixel_label(target)
        
        # Next we determine the IoU scores for each prediction and target pair.
        # The IoU score acts as the prediction score.
        iou_scores = self._iou(prediction_labels, target_labels)
        
        # If the IoU is higher than the given threshold, the prediction is considered correct.
        # (i.e. the prediction is a true positive)
        true_positive = iou_scores >= self._iou_threshold
        false_positive = iou_scores < self._iou_threshold
        
        precision = true_positive / (true_positive + false_positive)
        
        
        
        
    def _determine_per_pixel_label(t: torch.Tensor):
        probs = F.softmax(t, dim=1)
        predicted_class = torch.argmax(probs, dim=1, keepdims=True)
        class_per_channel = torch.zeros_like(t)
        class_per_channel.scatter_(1, predicted_class, 1)
        return class_per_channel

    
    def _iou(self, prediction: torch.Tensor, target: torch.Tensor):
        """
        Implements the intersection over union calculation.
        """
        prediction_bv = self._get_blood_vessel_channel(prediction)
        target_bv = self._get_blood_vessel_channel(target)
        
        intersection = torch.logical_and(prediction_bv, target_bv)
        union = torch.logical_or(prediction_bv, target_bv)
        iou_score = torch.sum(intersection, dim=(1, 2)) / torch.sum(union, dim=(1, 2))
        return iou_score
        
    def _get_blood_vessel_channel(self, t):
        return t[:, self._blood_vessel_channel, :, :]