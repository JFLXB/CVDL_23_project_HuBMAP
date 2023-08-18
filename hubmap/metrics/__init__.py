import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from checkpoints import CHECKPOINT_DIR
from configs import CONFIG_DIR
from hubmap.data import DATA_DIR
from hubmap.dataset import transforms as T
from hubmap.dataset import ValDataset, TestDataset
from hubmap.models.trans_res_u_net.model import TResUnet, TResUnet512

BLOOD_VESSEL_CLASS_INDEX = 0


class Precision:
    """
    Code based on: https://github.com/nikhilroxtomar/TransResUNet
    """

    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Precision"):
        self._name = name
    
    def __call__(self, prediction, target):
        intersection = (target * prediction).sum((-2, -1))
        result = (intersection + 1e-15) / (prediction.sum((-2, -1)) + 1e-15)
        return result[:, BLOOD_VESSEL_CLASS_INDEX]


class Recall:
    """
    Code based on: https://github.com/nikhilroxtomar/TransResUNet
    """

    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Recall"):
        self._name = name
    
    def __call__(self, prediction, target):
        intersection = (target * prediction).sum((-2, -1))
        result = (intersection + 1e-15) / (target.sum((-2, -1)) + 1e-15)
        return result[:, BLOOD_VESSEL_CLASS_INDEX]


class F2:
    """
    Code based on: https://github.com/nikhilroxtomar/TransResUNet
    """

    @property
    def name(self):
        return self._name
    
    def __init__(self, name="F2", beta=2):
        self._name = name
        self._beta = beta
    
    def __call__(self, prediction, target):
        p = Precision()(prediction, target)
        r = Recall()(prediction, target)
        return (1+self._beta**2.) *(p*r) / float(self._beta**2*p + r + 1e-15)


class DiceScore:
    """
    Code based on: https://github.com/nikhilroxtomar/TransResUNet
    """

    @property
    def name(self):
        return self._name
    
    def __init__(self, name="DiceScore"):
        self._name = name
    
    def __call__(self, prediction, target):
        result = (2 * (target * prediction).sum((-2, -1)) + 1e-15) / (target.sum((-2, -1)) + prediction.sum((-2, -1)) + 1e-15)
        return result[:, BLOOD_VESSEL_CLASS_INDEX]


class Jac:
    """
    Code based on: https://github.com/nikhilroxtomar/TransResUNet
    """
    
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Jac"):
        self._name = name
    
    def __call__(self, prediction, target):
        intersection = (target * prediction).sum((-2, -1))
        union = target.sum((-2, -1)) + prediction.sum((-2, -1)) - intersection
        jac = (intersection + 1e-15) / (union + 1e-15)
        return jac[:, BLOOD_VESSEL_CLASS_INDEX]


class Acc:
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Acc"):
        self._name = name
        
    def __call__(self, prediction, target):
        prediction_bv_mask = prediction[:, BLOOD_VESSEL_CLASS_INDEX, :, :]
        target_bv_mask = target[:, BLOOD_VESSEL_CLASS_INDEX, :, :]
        
        correct = (prediction_bv_mask == target_bv_mask).sum((-2, -1))
        total = target.size(-2) * target.size(-1)
        accuracy = correct / total
        return accuracy
    
    
class Confidence:
    @property
    def name(self):
        return self._name
    
    def __init__(self, name="Confidence"):
        self._name = name
        
    def __call__(self, probs):
        # Get the maximum probability over all classes
        max_probs, _ = probs.max(dim=1)
        # Select the blood vessel probabilities
        blood_vessel_probs = max_probs[:, BLOOD_VESSEL_CLASS_INDEX]
        # Calculate the mean probability over all pixels
        mean_prob = blood_vessel_probs.mean()
        return mean_prob


def calculate_statistics(model, device, val_set, val_loader, metrics):
    best_model_results = torch.zeros(len(val_set), len(metrics))
    model.eval()
    with torch.no_grad():
        for i, (image, mask) in enumerate(val_loader):
            image = image.to(device)
            mask = mask.to(device)
            
            prediction = model(image)
        
            probs = F.sigmoid(prediction)
            classes = torch.argmax(probs, dim=1, keepdims=True)
            classes_per_channel = torch.zeros_like(prediction)
            classes_per_channel.scatter_(1, classes, 1)
            
            for j, metric in enumerate(metrics):
                if isinstance(metric, Confidence):
                    best_model_results[i, j] = metric(probs)
                else:
                    best_model_results[i, j] = metric(classes_per_channel, mask)
    return best_model_results


def print_statistics(results, metrics, title):
    mean_results = results.mean(dim=0).numpy()

    assert len(mean_results) == len(metrics)

    metric_names = [metric.name for metric in metrics]
    print("-----------------------------------")
    print(title)
    for i, metric_name in enumerate(metric_names):
        print(f"\t{metric_name}: {mean_results[i]:.4f}")
    print("-----------------------------------")