import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCEDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
#         print(inputs.size(), targets.size())
        inputs = F.sigmoid(inputs)       
#         print(inputs.size(), targets.size())
        #flatten label and prediction tensors
#         print("do flatten")
#         inputs = inputs.view(-1)
        inputs = inputs.flatten()
#         print(inputs.size(), targets.size())
#         targets = targets.view(-1)
        targets = targets.flatten()
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE