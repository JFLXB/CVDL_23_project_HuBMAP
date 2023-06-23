from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader

from hubmap.models.dpt import DPT
from hubmap.losses.bce_dice_loss import BCEDiceLoss


train_dataset = VOCSegmentation(root="./data", image_set="train", download=True)
val_dataset = VOCSegmentation(root="./data", image_set="val", download=True)


dpt = DPT(num_classes=20)
