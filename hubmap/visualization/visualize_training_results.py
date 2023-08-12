from typing import Dict
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scienceplots as _

from checkpoints import CHECKPOINT_DIR


def visualize_checkpoint(checkpoint_name: str):
    plt.style.use(["science"])
    checkpoint = torch.load(Path(CHECKPOINT_DIR / checkpoint_name))

    # epoch = checkpoint["epoch"] + 1
    # start_epoch = checkpoint.get("start_epoch", 1)
    training_loss_history = checkpoint["training_loss_history"]
    training_metric_history = checkpoint["training_metric_history"]
    testing_loss_history = checkpoint["testing_loss_history"]
    testing_metric_history = checkpoint["testing_metric_history"]

    loss_figure = _create_figure(
        training_loss_history, testing_loss_history, "Loss", "Training and Testing Loss"
    )
    metric_figure = _create_figure(
        training_metric_history,
        testing_metric_history,
        "Benchmark",
        "Training and Testing Benchmark Values",
    )

    return loss_figure, metric_figure


def visualize_result(result: Dict):
    plt.style.use(["science"])
    data_train = result["training"]["loss"]
    data_test = result["testing"]["loss"]
    loss_figure = _create_figure(
        data_train, data_test, "Loss", "Training and Testing Loss"
    )

    data_train = result["training"]["metric"]
    data_test = result["testing"]["metric"]
    metric_figure = _create_figure(
        data_train, data_test, "Benchmark", "Training and Testing Benchmark Values"
    )

    return loss_figure, metric_figure


def _create_figure(data_train, data_test, y_label, title):
    data_train = _prepare_data(data_train)
    data_test = _prepare_data(data_test)

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


def _prepare_data(data):
    d = [(i, e) for i, elems in enumerate(data) for e in elems]
    df = pd.DataFrame(d, columns=["epoch", "value"])
    return df
