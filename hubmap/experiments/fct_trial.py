import torch
import torch.optim as optim

from hubmap.experiments.load_data import make_annotated_loader
from hubmap.dataset import transforms as T
from hubmap.losses import MultiOutputBCELoss
from hubmap.metrics import IoU
from hubmap.training import train
from hubmap.training import LRScheduler
# from hubmap.training import EarlyStopping

from hubmap.models import FCT
from hubmap.models.fct import init_weights


IMG_DIM = 512
# Batch size of 1 is 4.91GB of GPU memory for the FCT model.
BATCH_SIZE = 4 # MAXIMUM FOR THE IMAGE SIZE OF 512x512
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

checkpoint_name = "fct_trial.pt"

model = FCT(in_channels=3, num_classes=3).to(device)
model.apply(init_weights)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = MultiOutputBCELoss(weights=[0.14, 0.29, 0.57], interpolation_strategy="bilinear")

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
    loss_out_index=None,
    benchmark_out_index=2
)
