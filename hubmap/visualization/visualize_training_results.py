from typing import Dict, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scienceplots as _


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
