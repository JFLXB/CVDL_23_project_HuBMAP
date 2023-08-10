import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from hubmap.experiments.load_data import make_annotated_loader
from hubmap.dataset import transforms as T
from hubmap.losses import MultiOutputBCELoss
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
args = parser.parse_args()


IMG_DIM = args.image_size
# Batch size of 1 is 4.91GB of GPU memory for the FCT model.
# Batch size of 4 is the MAXIMUM FOR THE IMAGE SIZE OF 512x512
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs


train_transformations = T.Compose(
    [
        T.ToTensor(),
        T.Resize((IMG_DIM, IMG_DIM)),
        # T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(),
        # T.RandomCrop(size=(IMG_DIM, IMG_DIM)),
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

checkpoint_name = (
    f"fct_overfit_img_size_{IMG_DIM}.pt"
)

model = FCT(in_channels=3, num_classes=3).to(device)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# criterion = MultiOutputBCELoss(
    # weights=[0.14, 0.29, 0.57], interpolation_strategy="bilinear"
# )
criterion = nn.BCELoss()

# WE ARE ONLY INTERESTED IN THE IoU OF THE BLOOD VESSEL CLASS FOR NOW.
benchmark = IoU(class_index=0)
# train_loader, test_loader = load_annotated_data(BATCH_SIZE)
lr_scheduler = LRScheduler(optimizer, patience=20, min_lr=1e-6, factor=0.8)
# early_stopping = EarlyStopping(patience=50, min_delta=0.0)

# image, target = next(iter(train_loader))
# image = torch.zeros((1, 3, IMG_DIM, IMG_DIM))
# target = torch.zeros((1, 3, IMG_DIM, IMG_DIM))

# box_size = IMG_DIM // 4
# image[:, 0, 0:box_size, 0:box_size] = 1
# image[:, 0, box_size:2*box_size, box_size:2*box_size] = 1
# target[:, 0, 0:box_size, 0:box_size] = 1

image = torch.zeros((1, 3, IMG_DIM, IMG_DIM))
box_size = IMG_DIM // 4
image[:, 0, 0:box_size, 0:box_size] = 1
image[:, 1, box_size:2*box_size, box_size:2*box_size] = 1
image[:, 2, 2*box_size:3*box_size, 2*box_size:3*box_size] = 1
target = deepcopy(image)


# SAVE IMAGES
torch.save(image, "image.pt")
torch.save(target, "target.pt")

train_loader = [(image, target) for _ in range(BATCH_SIZE)]
test_loader = [(image, target) for _ in range(BATCH_SIZE)]

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
