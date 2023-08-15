import os
import argparse
from pathlib import Path
from configs import CONFIG_DIR
from figures import FIGURES_DIR

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from hubmap.data import DATA_DIR
from hubmap.dataset import transforms as T
from hubmap.dataset import TrainDataset, ValDataset

from hubmap.experiments.TransResUNet.utils import run
from hubmap.experiments.TransResUNet.utils import DiceBCELoss
from hubmap.experiments.TransResUNet.utils import visualize_detailed_results
from hubmap.experiments.TransResUNet.utils import visualize_detailed_results_overlay

from hubmap.training import LRScheduler
from hubmap.training import EarlyStopping

from hubmap.visualization import visualize_result

from hubmap.models.trans_res_u_net.model import TResUnet

NUM_EPOCHS = 200
BATCH_SIZE = 16
CHECKPOINT = "pretrained_resnet50_trial_1"
CONTINUE_TRAINING = False
PATIENCE = 20
LR = 1e-4
BACKBONE = "resnet50"
PRETRAINED = True

# FIGURES_CHECKPOINT_PATH = Path(FIGURES_DIR, "TransResUNet", f"{CHECKPOINT}")
# os.makedirs(FIGURES_CHECKPOINT_PATH, exist_ok=True)

# CHECKPOINT_FILE_NAME = f"{CHECKPOINT}.pt"
# CHECKPOINT_NAME = Path("TransResUNet", CHECKPOINT_FILE_NAME)
# config = {
#     "num_epochs": NUM_EPOCHS,
#     "batch_size": BATCH_SIZE,
#     "checkpoint_name": CHECKPOINT_NAME,
#     "patience": PATIENCE,
#     "lr": LR,
#     "backbone": BACKBONE,
#     "pretrained": PRETRAINED,
#     "figures_directory": FIGURES_CHECKPOINT_PATH
# }
# os.makedirs(Path(CONFIG_DIR / CHECKPOINT_NAME).parent.resolve(), exist_ok=True)
# torch.save(config, Path(CONFIG_DIR / CHECKPOINT_NAME))

# train_transforms = T.Compose(
#     [
#         T.ToTensor(),
#         T.Resize((256, 256)),
#         T.RandomHorizontalFlip(),
#         T.RandomVerticalFlip(),
#         T.RandomCrop((256, 256)),
#     ]
# )

# val_transforms = T.Compose(
#     [
#         T.ToTensor(),
#         T.Resize((256, 256)),
#     ]
# )

# train_set = TrainDataset(DATA_DIR, transform=train_transforms, with_background=True)
# val_set = ValDataset(DATA_DIR, transform=val_transforms, with_background=True)

# train_loader = DataLoader(
#     train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=16
# )
# val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

# device = "cuda" if torch.cuda.is_available() else "cpu"

# print(f"{CHECKPOINT} on {device}")

# model = TResUnet(num_classes=4, backbone=BACKBONE, pretrained=PRETRAINED)
# model = model.to(device)

# optimizer = optim.Adam(model.parameters(), lr=LR)
# criterion = DiceBCELoss()
# lr_scheduler = LRScheduler(optimizer, patience=PATIENCE)
# early_stopping = None

# result = run(
#     num_epochs=NUM_EPOCHS,
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     optimizer=optimizer,
#     criterion=criterion,
#     device=device,
#     early_stopping=early_stopping,
#     lr_scheduler=lr_scheduler,
#     checkpoint_name=CHECKPOINT_NAME,
#     continue_training=CONTINUE_TRAINING,
# )

# loss_fig, benchmark_fig = visualize_result(result)
# loss_fig.savefig(Path(FIGURES_CHECKPOINT_PATH, "results_loss.png"))
# benchmark_fig.savefig(Path(FIGURES_CHECKPOINT_PATH, "results_accuracy.png"))

# data = iter(val_set)
# image, target = next(data)
# detailed = visualize_detailed_results(model, image, target, device, CHECKPOINT_NAME)
# detailed.savefig(Path(FIGURES_CHECKPOINT_PATH, "example_results.png"))

# detailed_overlay = visualize_detailed_results_overlay(
#     model, image, target, device, CHECKPOINT_NAME
# )

# detailed_overlay.savefig(Path(FIGURES_CHECKPOINT_PATH, "example_overlay.png"))