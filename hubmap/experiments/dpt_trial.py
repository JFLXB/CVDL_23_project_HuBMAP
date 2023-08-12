import sys
import torch
import torch.optim as optim

# from hubmap.experiments.load_data import make_expert_loader
from hubmap.experiments.load_data import make_annotated_loader
from hubmap.dataset import transforms as T
from hubmap.losses import BCEDiceLoss
from hubmap.metrics import IoU
from hubmap.training import train
from hubmap.training import LRScheduler

# from hubmap.training import EarlyStopping

from hubmap.models import DPT
from hubmap.models.dpt import Backbone


use_pretrained = sys.argv[1] == "True"
backbone = Backbone(sys.argv[2])


IMG_DIM = 128
BATCH_SIZE = 64
NUM_EPOCHS = 200


train_transformations = T.Compose(
    [
        T.ToTensor(),
        T.Resize((IMG_DIM, IMG_DIM)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomCrop(size=(IMG_DIM, IMG_DIM)),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

test_transformations = T.Compose(
    [
        T.ToTensor(),
        T.Resize((IMG_DIM, IMG_DIM)),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

load_annotated_data = make_annotated_loader(train_transformations, test_transformations)
device = "cuda" if torch.cuda.is_available() else "cpu"


backbones = [Backbone.vitb16_384, Backbone.vitb_rn50_384, Backbone.vitl16_384]
pretrained = [False, True]

# for use_pretrained in pretrained:
# for backbone in backbones:
# suffix = "pretrained" if use_pretrained else "scratch"
# checkpoint_name = f"dpt_{backbone.value}_{suffix}"

# model = DPT(
#     num_classes=3, backbone=backbone, use_pretrained=use_pretrained).to(device)

# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# criterion = BCEDiceLoss()
# # WE ARE ONLY INTERESTED IN THE IoU OF THE BLOOD VESSEL CLASS FOR NOW.
# benchmark = IoU(class_index=0)
# train_loader, test_loader = load_annotated_data(BATCH_SIZE)
# lr_scheduler = LRScheduler(optimizer, patience=20, min_lr=1e-6, factor=0.8)
# # early_stopping = EarlyStopping(patience=50, min_delta=0.0)
# result = train(
#     num_epochs=NUM_EPOCHS,
#     model=model,
#     optimizer=optimizer,
#     criterion=criterion,
#     train_loader=train_loader,
#     test_loader=test_loader,
#     device=device,
#     benchmark=benchmark,
#     checkpoint_name=checkpoint_name,
#     learning_rate_scheduler=lr_scheduler,
# )


suffix = "pretrained" if use_pretrained else "scratch"
checkpoint_name = f"dpt_{backbone.value}_{suffix}"

model = DPT(num_classes=3, backbone=backbone, use_pretrained=use_pretrained).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = BCEDiceLoss()
# WE ARE ONLY INTERESTED IN THE IoU OF THE BLOOD VESSEL CLASS FOR NOW.
benchmark = IoU(class_index=0)
train_loader, test_loader = load_annotated_data(BATCH_SIZE)
lr_scheduler = LRScheduler(optimizer, patience=20, min_lr=1e-6, factor=0.8)
# early_stopping = EarlyStopping(patience=50, min_delta=0.0)
result = train(
    num_epochs=NUM_EPOCHS,
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    benchmark=benchmark,
    checkpoint_name=checkpoint_name,
    learning_rate_scheduler=lr_scheduler,
)
