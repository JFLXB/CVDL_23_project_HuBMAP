# %% [markdown]
# # Experiment Pipeline

# %%
import torch
import torch.nn as nn
import torch.optim as optim

from hubmap.experiments.load_data import make_expert_loader
from hubmap.experiments.load_data import make_annotated_loader
from hubmap.experiments.training import train
from hubmap.dataset import transforms as T
from hubmap.losses import BCEDiceLoss
from hubmap.metrics import MeanAveragePrecision

# %% [markdown]
# First we load the data that we need for the experiment. This includes the training data, the validation (test) data that we will use for training.
# 
# For this, depending on the experiments we use different transformations on the data. The following transformations are a minimal example. Furhter transformations should be included for more sophisticated experiments.

# %%
train_transformations = T.Compose(
    [T.ToTensor(), T.Resize((512, 512)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

test_transformations = T.Compose(
    [T.ToTensor(), T.Resize((512, 512)), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# %% [markdown]
# Depending on the experiment we may want to load all annotated images or just the ones that are annotated by experts.
# 
# Here we create a function to load all the images that are annotated (not only the ones by experts).
# The created function can than be used to load the data loaders with a specific batch size.

# %%
# The train, test split ratio is set to 0.8 by default.
# Meaning 80% of the data is used for training and 20% for testing.
load_annotated_data = make_annotated_loader(train_transformations, test_transformations)

# %% [markdown]
# In the following, we determine the device we want to train on. 
# If a GPU is available, we use it, otherwise we fall back to the CPU. 
# We also set the random seed for reproducibility.

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# Next, we need to load the model we want to train.

# %%
from hubmap.models import DPT
model = DPT(num_classes=3, features=128).to(device)

# %% [markdown]
# Next we create the other modules needed for training, such as the loss measure, and the optimizer.

# %%
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = BCEDiceLoss()
# metric = MeanAveragePrecision() # TODO: implement the mean Average Precision metric

# %% [markdown]
# Next, we initialize the trainer and start training. The trainer is responsible for running the training loop, saving checkpoints, and logging metrics 

# %%
BATCH_SIZE = 1

train_loader, test_loader = load_annotated_data(BATCH_SIZE)

# %%
result = train(
    num_epochs=1,
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    metric=None,
    checkpoint_name="dpt_trials",
)

# %% [markdown]
# Now we can visualize the results.
# (*this needs improvements + better and more visualizations for the final paper*)

# %%
from hubmap.experiments.visualization import visualize_result

# %%
visualize_result(result)

# %%



