import os
import argparse
import time
from pathlib import Path
from configs import CONFIG_DIR
from figures import FIGURES_DIR
from checkpoints import CHECKPOINT_DIR

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from hubmap.data import DATA_DIR
from hubmap.dataset import transforms as T
from hubmap.dataset import TrainDataset, ValDataset

from hubmap.experiments.TransResUNet.utils import run
from hubmap.experiments.TransResUNet.utils import DiceBCELoss, ChannelWeightedDiceBCELoss
from hubmap.experiments.TransResUNet.statistics import Precision, Recall, F2, DiceScore, Jac, Acc, Confidence, calculate_statistics, print_statistics

from hubmap.training import LRScheduler
from hubmap.training import EarlyStopping

from hubmap.visualization import visualize_result

from hubmap.models.trans_res_u_net.model import TResUnet, TResUnet512


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--epochs", type=int, required=True)
parser.add_argument("--backbone", type=str, required=True)
parser.add_argument("--pretrained", action='store_true')
parser.add_argument("--model", type=str, required=True)

# Optional
# LRScheduler
parser.add_argument("--use-lr-scheduler", action='store_true')
parser.add_argument("--lrs-patience", type=str, required=False, default=None)
# EarlyStopping
parser.add_argument("--use-early-stopping", action='store_true')
parser.add_argument("--es-patience", type=str, required=False, default=None)
# Loss
parser.add_argument("--loss", type=str, required=False, default="DiceBCELoss")
parser.add_argument("--weights", nargs="+", type=int, required=False, default=None)
args = parser.parse_args()

print(args)

MODEL = None
BATCH_SIZE = None
IMG_SIZE = None
if args.model == "TransResUNet":
    MODEL = TResUnet
    BATCH_SIZE = 16
    IMG_SIZE = 256
elif args.model == "TransResUNet512":
    MODEL = TResUnet512
    BATCH_SIZE = 8
    IMG_SIZE = 512
else:
    raise ValueError(f"Unknown model {args.model}")

LOSS = None
WEIGHT = None
if args.loss == "DiceBCELoss":
    LOSS = DiceBCELoss
elif args.loss == "ChannelWeightedDiceBCELoss":
    LOSS = ChannelWeightedDiceBCELoss
    WEIGHT = torch.tensor(args.weights)
    
LR_SCHEDULER = None
LRS_PATIENCE = None
if args.use_lr_scheduler:
    LRS_PATIENCE = args.lrs_patience
    LR_SCHEDULER = LRScheduler

EARLY_STOPPING = None
ES_PATIENCE = None
if args.use_early_stopping:
    ES_PATIENCE = args.es_patience
    EARLY_STOPPING = EarlyStopping

CHECKPOINT = args.name
NUM_EPOCHS = args.epochs
BACKBONE = args.backbone
PRETRAINED = args.pretrained

LR = 1e-4
CONTINUE_TRAINING = False

FIGURES_CHECKPOINT_PATH = Path(FIGURES_DIR, "TransResUNet", f"{CHECKPOINT}")
os.makedirs(FIGURES_CHECKPOINT_PATH, exist_ok=True)

CHECKPOINT_FILE_NAME = f"{CHECKPOINT}.pt"
CHECKPOINT_NAME = Path("TransResUNet", CHECKPOINT_FILE_NAME)
config = {
    "model": args.model,
    "image_size": IMG_SIZE,
    "num_epochs": NUM_EPOCHS,
    "batch_size": BATCH_SIZE,
    "checkpoint_name": CHECKPOINT_NAME,
    "use_lr_scheduler": LR_SCHEDULER is not None,
    "lr_scheduler_patience": LRS_PATIENCE,
    "use_early_stopping": EARLY_STOPPING is not None,
    "early_stopping_patience": ES_PATIENCE,
    "loss": args.loss,
    "lr": LR,
    "backbone": BACKBONE,
    "pretrained": PRETRAINED,
    "figures_directory": FIGURES_CHECKPOINT_PATH,
    "weight": WEIGHT
}
os.makedirs(Path(CONFIG_DIR / CHECKPOINT_NAME).parent.resolve(), exist_ok=True)
torch.save(config, Path(CONFIG_DIR / CHECKPOINT_NAME))

train_transforms = T.Compose(
    [
        T.ToTensor(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomCrop((IMG_SIZE, IMG_SIZE)),
    ]
)

val_transforms = T.Compose(
    [
        T.ToTensor(),
        T.Resize((IMG_SIZE, IMG_SIZE)),
    ]
)

train_set = TrainDataset(DATA_DIR, transform=train_transforms, with_background=True)
val_set = ValDataset(DATA_DIR, transform=val_transforms, with_background=True)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=16
)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MODEL(num_classes=4, backbone=BACKBONE, pretrained=PRETRAINED)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)

criterion = LOSS(weights=WEIGHT)
lr_scheduler = LR_SCHEDULER(optimizer, patience=LRS_PATIENCE) if LR_SCHEDULER is not None else None
early_stopping = EARLY_STOPPING(patience=ES_PATIENCE) if EARLY_STOPPING is not None else None

start = time.time()
result = run(
    num_epochs=NUM_EPOCHS,
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    early_stopping=early_stopping,
    lr_scheduler=lr_scheduler,
    checkpoint_name=CHECKPOINT_NAME,
    continue_training=CONTINUE_TRAINING,
)
total = time.time() - start

print(f"TRAINING TOOK A TOTAL OF '{total}' FOR '{NUM_EPOCHS}' EPOCHS => PER EPOCH TIME: '{total / NUM_EPOCHS}'")

loss_fig, benchmark_fig = visualize_result(result)
loss_fig.savefig(Path(FIGURES_CHECKPOINT_PATH, "results_loss.png"))
benchmark_fig.savefig(Path(FIGURES_CHECKPOINT_PATH, "results_accuracy.png"))


# STATISTICS
val_set = ValDataset(DATA_DIR, transform=val_transforms, with_background=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=16)
metrics = [Precision(), Recall(), F2(), DiceScore(), Jac(), Acc(), Confidence()]
# STATISTICS FOR LAST MODEL
best_model_results = calculate_statistics(model, device, val_set, val_loader, metrics)
print_statistics(best_model_results, metrics, "LAST MODEL STATISTICS")
# STATISTICS FOR BEST MODEL
parent = CHECKPOINT_NAME.parent
best_checkpoint_name = parent / f"{CHECKPOINT_NAME.stem}_best{CHECKPOINT_NAME.suffix}"
best_checkpoint_path = Path(CHECKPOINT_DIR / best_checkpoint_name)
best_model = MODEL(num_classes=4, backbone=BACKBONE, pretrained=PRETRAINED)
best_model_checkpoint = torch.load(best_checkpoint_path)
best_model.load_state_dict(best_model_checkpoint["model_state_dict"])
best_model = best_model.to(device)

best_model_results = calculate_statistics(best_model, device, val_set, val_loader, metrics)
print_statistics(best_model_results, metrics, "BEST MODEL STATISTICS")