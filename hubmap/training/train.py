from typing import Optional
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path

from hubmap.training.learning_rate_scheduler import LRScheduler
from hubmap.training.early_stopping import EarlyStopping

from checkpoints import CHECKPOINT_DIR


def train(
    num_epochs: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    benchmark: nn.Module,  # TODO: allow multiple benchmarks.
    checkpoint_name: str,
    learning_rate_scheduler: Optional[LRScheduler] = None,
    early_stopping: Optional[EarlyStopping] = None,
    continue_training: bool = False,
):
    """_summary_
    
    If the continue_training flag is set to True the checkpoint will be 
    loaded from the checkpoint directory and training is continued from the last
    checkpoint. And all information and progress up to the checkpoint will be loaded.

    Parameters
    ----------
    num_epochs : int
        _description_
    model : nn.Module
        _description_
    optimizer : torch.optim.Optimizer
        _description_
    criterion : nn.Module
        _description_
    train_loader : DataLoader
        _description_
    test_loader : DataLoader
        _description_
    device : str
        _description_
    benchmark : nn.Module
        _description_
    learning_rate_scheduler : Optional[LRScheduler], optional
        _description_, by default None
    early_stopping : Optional[EarlyStopping], optional
        _description_, by default None
    continue_training : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    start_epoch = 1

    training_loss_history = []
    training_metric_history = []

    testing_loss_history = []
    testing_metric_history = []

    # CHECK start_epoch.
    if continue_training:
        # Load checkpoint.
        checkpoint = torch.load(Path(CHECKPOINT_DIR / checkpoint_name))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        training_loss_history = checkpoint["training_loss_history"]
        training_metric_history = checkpoint["training_metric_history"]
        testing_loss_history = checkpoint["testing_loss_history"]
        testing_metric_history = checkpoint["testing_metric_history"]


    for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
        tqdm.write(f"Epoch {epoch}/{num_epochs} - Started training...")
        training_losses = []
        training_accuracies = []
        model.train()
        for images, targets in tqdm(train_loader, leave=False):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(images)

            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            metric = benchmark(predictions, targets)

            training_losses.append(loss.item())
            training_accuracies.append(metric.item())

        training_loss_history.append(training_losses)
        training_metric_history.append(training_accuracies)

        tqdm.write(f"Epoch {epoch}/{num_epochs} - Started testing...")
        testing_losses = []
        testing_accuracies = []
        model.eval()
        for images, targets in tqdm(test_loader, leave=False):
            images = images.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                predictions = model(images)
                loss = criterion(predictions, targets)
                metric = benchmark(predictions, targets)

            testing_losses.append(loss.item())
            testing_accuracies.append(metric.item())

        testing_loss_history.append(testing_losses)
        testing_metric_history.append(testing_accuracies)

        tqdm.write(
            f"Epoch {epoch}/{num_epochs} - Summary:\n"
            f"\tTraining Loss: {np.mean(training_losses)}\n"
            f"\tTraining metric: {np.mean(training_accuracies)}\n"
            f"\tTesting Loss: {np.mean(testing_losses)}\n"
            f"\tTesting metric: {np.mean(testing_accuracies)}\n"
        )

        data_to_save = {
            "early_stopping": False,
            "epoch": epoch,
            # "start_epoch": start_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_loss_history": training_loss_history,
            "training_metric_history": training_metric_history,
            "testing_loss_history": testing_loss_history,
            "testing_metric_history": testing_metric_history,
        }

        # NOW DO THE ADJUSTMENTS USING THE LEARNING RATE SCHEDULER.
        if learning_rate_scheduler:
            learning_rate_scheduler(np.mean(testing_losses))
        # NOW DO THE ADJUSTMENTS USING THE EARLY STOPPING.
        if early_stopping:
            early_stopping(np.mean(testing_losses))
            # MODIFY THE DATA TO SAVE ACCORDING TO THE EARLY STOPPING RESULT.
            data_to_save["early_stopping"] = early_stopping.early_stop

        # SAVE THE DATA.
        torch.save(data_to_save, Path(CHECKPOINT_DIR / checkpoint_name))

        # DO THE EARLY STOPPING IF NECESSARY.
        if early_stopping and early_stopping.early_stop:
            break

    result = {
        "epoch": epoch,
        "training": {
            "loss": training_loss_history,
            "metric": training_metric_history,
        },
        "testing": {
            "loss": testing_loss_history,
            "metric": testing_metric_history,
        },
    }
    return result
