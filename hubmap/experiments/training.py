import os
import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from pathlib import Path


def train(
    num_epochs: int,
    model: nn.Module, 
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: str,
    metric: nn.Module,
    checkpoint_name: str,
    start_epoch: int = 1,
):
    """_summary_

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
    metric : nn.Module
        _description_
    start_epoch : int, optional
        _description_, by default 1

    Returns
    -------
    _type_
        _description_
    """
    training_loss_history = []
    training_metric_history = []

    testing_loss_history = []
    testing_metric_history = []

    for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
        tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Started training...")
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

            metric = _calculate_metric(predictions, targets)

            training_losses.append(loss.item())
            training_accuracies.append(metric.item())

        training_loss_history.append(training_losses)
        training_metric_history.append(training_accuracies)

        tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Started testing...")
        testing_losses = []
        testing_accuracies = []
        model.eval()
        for images, targets in tqdm(test_loader, leave=False):
            images = images.to(device)
            targets = targets.to(device)

            with torch.no_grad():
                predictions = model(images)
                loss = criterion(predictions, targets)
                metric = _calculate_metric(predictions, targets)

            testing_losses.append(loss.item())
            testing_accuracies.append(metric.item())

        testing_loss_history.append(testing_losses)
        testing_metric_history.append(testing_accuracies)

        tqdm.write(
            f"Epoch {epoch + 1}/{num_epochs} - Summary:\n"
            f"\tTraining Loss: {np.mean(training_losses)}\n"
            f"\tTraining metric: {np.mean(training_accuracies)}\n"
            f"\tTesting Loss: {np.mean(testing_losses)}\n"
            f"\tTesting metric: {np.mean(testing_accuracies)}\n"
        )
        
        curr_dir = Path(__file__).parent.absolute()
        checkpoint_dir = Path(curr_dir / "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_loss_history': training_loss_history,
            'training_metric_history': training_metric_history,
            'testing_loss_history': testing_loss_history,
            'testing_metric_history': testing_metric_history,
        }, Path(checkpoint_dir / checkpoint_name))

    result = {
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

def _calculate_metric(predictions, targets):
    # We only want to determine the accuracy for the blood_vessel class.
    return torch.tensor(0.0)




# class Trainer:
#     def __init__(
#         self,
#         model: nn.Module,
#         optimizer: torch.optim.Optimizer,
#         criterion: nn.Module,
#         train_loader: DataLoader,
#         test_loader: DataLoader,
#         device: str,
#         metric: nn.Module
#     ):
#         self._model = model
#         self._train_loader = train_loader
#         self._test_loader = test_loader

#         self._optimizer = optimizer
#         self._criterion = criterion
#         self._device = device
#         self._last_result = {}
#         # TODO: Upgrade to support multiple metrics.
#         self._metric = metric

#     def execute(self, num_epochs: int):
#         training_loss_history = []
#         training_metric_history = []

#         testing_loss_history = []
#         testing_metric_history = []

#         for epoch in tqdm(range(num_epochs), leave=False):
#             tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Started training...")
#             training_result = self._train_epoch()
#             training_losses = []
#             training_accuracies = []

#             self._model.train()
#             for images, targets in tqdm(self._train_loader, leave=False):
#                 images = images.to(self._device)
#                 targets = targets.to(self._device)

#                 optimizer.zero_grad()
#                 predictions = model(images)

#                 loss = criterion(predictions, targets)
#                 loss.backward()
#                 optimizer.step()

#                 metric = calculate_metric(predictions, targets)

#                 losses.append(loss.item())
#                 accuracies.append(metric.item())

#             training_loss_history.append(training_result["loss"])
#             training_metric_history.append(training_result["metric"])

#             tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Started testing...")
#             testing_result = self._test_epoch()
#             testing_loss_history.append(testing_result["loss"])
#             testing_metric_history.append(testing_result["metric"])

#             tqdm.write(
#                 f"Epoch {epoch + 1}/{num_epochs} - Summary:\n"
#                 f"\tTraining Loss: {np.mean(training_result['loss'])}\n"
#                 f"\tTraining metric: {np.mean(training_result['metric'])}\n"
#                 f"\tTesting Loss: {np.mean(testing_result['loss'])}\n"
#                 f"\tTesting metric: {np.mean(testing_result['metric'])}\n"
#             )

#         self._last_result = {
#             "training": {
#                 "loss": training_loss_history,
#                 "metric": training_metric_history,
#             },
#             "testing": {
#                 "loss": testing_loss_history,
#                 "metric": testing_metric_history,
#             },
#         }
#         return self._last_result



#     def _train_epoch(self):
#         losses = []
#         accuracies = []

#         self._model.train()
#         for images, targets in tqdm(self._train_loader, leave=False):
#             images = images.to(self._device)
#             targets = targets.to(self._device)

#             self._optimizer.zero_grad()
#             predictions = self._model(images)

#             loss = self._criterion(predictions, targets)
#             loss.backward()
#             self._optimizer.step()

#             metric = self._calculate_metric(predictions, targets)

#             losses.append(loss.item())
#             accuracies.append(metric.item())

#         return {"loss": losses, "metric": accuracies}

#     def _test_epoch(self):
#         losses = []
#         accuracies = []

#         self._model.eval()
#         for images, targets in tqdm(self._test_loader, leave=False):
#             images = images.to(self._device)
#             targets = targets.to(self._device)

#             with torch.no_grad():
#                 predictions = self._model(images)
#                 loss = self._criterion(predictions, targets)
#                 metric = self._calculate_metric(predictions, targets)

#             losses.append(loss.item())
#             accuracies.append(metric.item())

#         return {"loss": losses, "metric": accuracies}

