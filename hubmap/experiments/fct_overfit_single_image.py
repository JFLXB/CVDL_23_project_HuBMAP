import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from hubmap.experiments.load_data import make_annotated_loader
from hubmap.dataset import transforms as T
from hubmap.losses import MultiOutputBCELoss
from hubmap.losses import BCEDiceLoss
from hubmap.losses import ClassWeightedBCELoss
from hubmap.metrics import IoU
from hubmap.training import train
from hubmap.training import LRScheduler

# from hubmap.training import EarlyStopping

from hubmap.models import FCT
from hubmap.models.fct import init_weights

parser = argparse.ArgumentParser()
parser.add_argument(
    "-b",
    "--batch-size",
    type=int,
    default=1,
    help="input batch size for training (default: 1)",
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    default=200,
    help="number of epochs to train (default: 200)",
)
parser.add_argument(
    "-i", "--image-size", type=int, default=256, help="image size (default: 256)"
)
parser.add_argument(
    "--add-background",
    action="store_true",
    help="Add background class to the target mask",
)
args = parser.parse_args()


IMG_DIM = args.image_size
# Batch size of 1 is 4.91GB of GPU memory for the FCT model.
# Batch size of 4 is the MAXIMUM FOR THE IMAGE SIZE OF 512x512
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs

WITH_BACKGROUND = args.add_background
NUM_CLASSES = 4 if WITH_BACKGROUND else 3


transformations = T.Compose(
    [
        T.ToTensor(),
        T.Resize((IMG_DIM, IMG_DIM)),
        # T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# load_annotated_data = make_annotated_loader(
#     train_transformations, test_transformations, with_background=WITH_BACKGROUND
# )
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint_name = f"fct_overfit_img_size_{args.image_size}.pt"

model = FCT(in_channels=3, num_classes=4).to(device)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# criterion = MultiOutputBCELoss(
#     weights=[0.14, 0.29, 0.57], interpolation_strategy="bilinear"
# )
# criterion = BCEDiceLoss()
criterion = nn.BCELoss()
# blood_vessel_weight = 3 / 4
# other_weight = (1 / 4) / 3
# criterion = ClassWeightedBCELoss(
#     weights=[blood_vessel_weight, other_weight, other_weight, other_weight]
# )

# WE ARE ONLY INTERESTED IN THE IoU OF THE BLOOD VESSEL CLASS FOR NOW.
# benchmark = IoU(class_index=0)
benchmark = IoU()
# train_loader, test_loader = load_annotated_data(BATCH_SIZE)
lr_scheduler = LRScheduler(optimizer, patience=20, min_lr=1e-8, factor=0.5)
# early_stopping = EarlyStopping(patience=50, min_delta=0.0)

from hubmap.dataset import BaseDataset
from hubmap.data import DATA_DIR

dataset = BaseDataset(DATA_DIR, transform=transformations, with_background=True)
image, target = dataset[3]
image, target = image.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
# target = target[:, :1, :, :]
# print(target.size())

train_loader = [(image, target)]
test_loader = [(image, target)]

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
    loss_out_index=2,
    benchmark_out_index=2,
)
