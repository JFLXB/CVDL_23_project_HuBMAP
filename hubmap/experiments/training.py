from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scienceplots as _

from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str,
        metric: nn.Module
    ):
        self._model = model
        self._train_loader = train_loader
        self._test_loader = test_loader

        self._optimizer = optimizer
        self._criterion = criterion
        self._device = device
        self._last_result = {}
        # TODO: Upgrade to support multiple metrics.
        self._metric = metric

    def execute(self, num_epochs: int):
        training_loss_history = []
        training_metric_history = []

        testing_loss_history = []
        testing_metric_history = []

        for epoch in tqdm(range(num_epochs), leave=False):
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Started training...")
            training_result = self._train_epoch()
            training_loss_history.append(training_result["loss"])
            training_metric_history.append(training_result["metric"])

            tqdm.write(f"Epoch {epoch + 1}/{num_epochs} - Started testing...")
            testing_result = self._test_epoch()
            testing_loss_history.append(testing_result["loss"])
            testing_metric_history.append(testing_result["metric"])

            tqdm.write(
                f"Epoch {epoch + 1}/{num_epochs} - Summary:\n"
                f"\tTraining Loss: {np.mean(training_result['loss'])}\n"
                f"\tTraining metric: {np.mean(training_result['metric'])}\n"
                f"\tTesting Loss: {np.mean(testing_result['loss'])}\n"
                f"\tTesting metric: {np.mean(testing_result['metric'])}\n"
            )

        self._last_result = {
            "training": {
                "loss": training_loss_history,
                "metric": training_metric_history,
            },
            "testing": {
                "loss": testing_loss_history,
                "metric": testing_metric_history,
            },
        }
        return self._last_result

    def visualise_result(self, result: Optional[Dict] = None):
        plt.style.use(["science"])

        result = result or self._last_result

        data_train = result["training"]["loss"]
        data_test = result["testing"]["loss"]
        loss_figure = self._create_figure(
            data_train, data_test, "Loss", "Training and Testing Loss"
        )

        data_train = result["training"]["metric"]
        data_test = result["testing"]["metric"]
        metric_figure = self._create_figure(
            data_train, data_test, "Metric", "Training and Testing Metric"
        )

        return loss_figure, metric_figure

    def _create_figure(self, data_train, data_test, y_label, title):
        data_train = self._prepare_data(data_train)
        data_test = self._prepare_data(data_test)

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        axs.grid()
        sns.lineplot(
            data_train,
            x="epoch",
            y="value",
            ax=axs,
            linestyle="solid",
            label="Training",
        )
        sns.lineplot(
            data_test, x="epoch", y="value", ax=axs, linestyle="dashed", label="Testing"
        )
        axs.set_xlabel("Epochs")
        axs.set_ylabel(y_label)
        axs.set_title(title)
        return fig

    def _prepare_data(self, data):
        d = [(i, e) for i, elems in enumerate(data) for e in elems]
        df = pd.DataFrame(d, columns=["epoch", "value"])
        return df

    def _train_epoch(self):
        losses = []
        accuracies = []

        self._model.train()
        for images, targets in tqdm(self._train_loader, leave=False):
            images = images.to(self._device)
            targets = targets.to(self._device)

            self._optimizer.zero_grad()
            predictions = self._model(images)

            loss = self._criterion(predictions, targets)
            loss.backward()
            self._optimizer.step()

            metric = self._calculate_metric(predictions, targets)

            losses.append(loss.item())
            accuracies.append(metric.item())

        return {"loss": losses, "metric": accuracies}

    def _test_epoch(self):
        losses = []
        accuracies = []

        self._model.eval()
        for images, targets in tqdm(self._test_loader, leave=False):
            images = images.to(self._device)
            targets = targets.to(self._device)

            with torch.no_grad():
                predictions = self._model(images)
                loss = self._criterion(predictions, targets)
                metric = self._calculate_metric(predictions, targets)

            losses.append(loss.item())
            accuracies.append(metric.item())

        return {"loss": losses, "metric": accuracies}

    def _calculate_metric(self, predictions, targets):
        # We only want to determine the accuracy for the blood_vessel class.
        pass
