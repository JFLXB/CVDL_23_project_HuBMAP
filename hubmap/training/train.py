from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    val_loader: DataLoader,
    device: str,
    benchmark: nn.Module,  # TODO: allow multiple benchmarks.
    checkpoint_name: str,
    learning_rate_scheduler: Optional[LRScheduler] = None,
    early_stopping: Optional[EarlyStopping] = None,
    continue_training: bool = False,
    # this is needed to select the output of the model to be used for the loss function
    # used for models that have multiple outputs, e.g. FCT
    loss_out_index: int = None,
    benchmark_out_index: int = None,
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
    # if isinstance(criterion, nn.Module):
    # criterion = [criterion]

    start_epoch = 1

    training_loss_history = []
    training_metric_history = []

    validation_loss_history = []
    validation_metric_history = []

    # CHECK start_epoch.
    if continue_training:
        # Load checkpoint.
        checkpoint = torch.load(Path(CHECKPOINT_DIR / checkpoint_name))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        training_loss_history = checkpoint["training_loss_history"]
        training_metric_history = checkpoint["training_metric_history"]
        validation_loss_history = checkpoint["validation_loss_history"]
        validation_metric_history = checkpoint["validation_metric_history"]

    for epoch in tqdm(range(start_epoch, start_epoch + num_epochs)):
        # tqdm.write(f"Epoch {epoch}/{num_epochs} - Started training...")
        training_losses = []
        training_accuracies = []
        model.train()
        for images, targets in tqdm(train_loader, leave=False):
            images = images.to(device)
            targets = targets.to(device)
            # targets_bv = targets[:, 0:1, :, :]
            # targets_bg = 1.0 - targets_bv
            # targets = torch.zeros((targets.size(0), 2, targets.size(2), targets.size(3)))
            # # print(targets_bv.size(), targets.size())
            # targets[:, 0, :, :] = targets_bv[:, 0, :, :]
            # targets[:, 1, :, :] = targets_bg[:, 0, :, :]
            # targets = targets.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            # print(predictions[2].size())
            # print(predictions[2].sigmoid().size())
            # assert False

            preds_for_loss = (
                predictions[loss_out_index]
                if loss_out_index is not None
                else predictions
            )
            # preds_for_benchmark = (
            #     predictions[benchmark_out_index]
            #     if benchmark_out_index is not None
            #     else predictions
            # )
            # preds_for_benchmark = F.softmax(preds_for_benchmark, dim=1)
            # classes = torch.argmax(preds_for_benchmark, dim=1, keepdim=True)
            # classes_per_channel = torch.zeros_like(preds_for_benchmark)
            # classes_per_channel = classes_per_channel.scatter_(1, classes, 1)

            # target_classes = torch.argmax(targets, dim=1)
            # print(targets.size(), preds_for_loss[0].size()[2:])

            t1 = (
                F.interpolate(
                    targets, size=preds_for_loss[0].size()[2:], mode="nearest"
                )
                .squeeze(1)
                .type(torch.LongTensor)
                .to(device)
            )
            t2 = (
                F.interpolate(
                    targets, size=preds_for_loss[1].size()[2:], mode="nearest"
                )
                .squeeze(1)
                .type(torch.LongTensor)
                .to(device)
            )
            t_final = targets.squeeze(1).type(torch.LongTensor).to(device)

            # preds_for_loss = F.softmax(preds_for_loss, dim=1)
            loss1 = criterion(preds_for_loss[0], t1)
            loss2 = criterion(preds_for_loss[1], t2)
            loss_final = criterion(preds_for_loss[2], t_final)

            loss1.backward(retain_graph=True)
            loss2.backward(retain_graph=True)
            loss_final.backward(retain_graph=True)

            optimizer.step()

            # metric = benchmark(classes_per_channel, targets)
            metric = torch.tensor(-1.0)

            loss = loss1 + loss2 + loss_final
            training_losses.append(loss.item())
            training_accuracies.append(metric.item())

        training_loss_history.append(training_losses)
        training_metric_history.append(training_accuracies)

        # tqdm.write(f"Epoch {epoch}/{num_epochs} - Started validation...")
        validation_losses = []
        validation_accuracies = []
        model.eval()
        for images, targets in tqdm(val_loader, leave=False):
            images = images.to(device)
            targets = targets.to(device)
            # targets_bv = targets[:, 0:1, :, :]
            # targets_bg = 1.0 - targets_bv
            # targets = torch.zeros((targets.size(0), 2, targets.size(2), targets.size(3)))
            # targets[:, 0, :, :] = targets_bv[:, 0, :, :]
            # targets[:, 1, :, :] = targets_bg[:, 0, :, :]
            # targets = targets.to(device)

            with torch.no_grad():
                predictions = model(images)
                preds_for_loss = (
                    predictions[loss_out_index]
                    if loss_out_index is not None
                    else predictions
                )
                # preds_for_benchmark = (
                #     predictions[benchmark_out_index]
                #     if benchmark_out_index is not None
                #     else predictions
                # )
                # preds_for_benchmark = F.softmax(preds_for_benchmark, dim=1)
                # classes = torch.argmax(preds_for_benchmark, dim=1, keepdim=True)
                # classes_per_channel = torch.zeros_like(preds_for_benchmark)
                # classes_per_channel = classes_per_channel.scatter_(1, classes, 1)

                # preds_for_loss = F.softmax(preds_for_loss, dim=1)

                t1 = (
                    F.interpolate(
                        targets, size=preds_for_loss[0].size()[2:], mode="nearest"
                    )
                    .squeeze(1)
                    .type(torch.LongTensor)
                    .to(device)
                )
                t2 = (
                    F.interpolate(
                        targets, size=preds_for_loss[1].size()[2:], mode="nearest"
                    )
                    .squeeze(1)
                    .type(torch.LongTensor)
                    .to(device)
                )
                t_final = targets.squeeze(1).type(torch.LongTensor).to(device)

                # preds_for_loss = F.softmax(preds_for_loss, dim=1)
                loss1 = criterion(preds_for_loss[0], t1)
                loss2 = criterion(preds_for_loss[1], t2)
                loss_final = criterion(preds_for_loss[2], t_final)

                loss = loss1 + loss2 + loss_final
                # metric = benchmark(classes_per_channel, targets)
                metric = torch.tensor(-1.0)

            validation_losses.append(loss.item())
            validation_accuracies.append(metric.item())

        validation_loss_history.append(validation_losses)
        validation_metric_history.append(validation_accuracies)

        tqdm.write(
            f"Epoch {epoch}/{num_epochs} - Summary: "
            f"TL: {np.mean(training_losses):.5f} "
            f"TB: {np.mean(training_accuracies):.5f} "
            f"VL: {np.mean(validation_losses):.5f} "
            f"VB: {np.mean(validation_accuracies):.5f}"
        )

        data_to_save = {
            "early_stopping": False,
            "epoch": epoch,
            # "start_epoch": start_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_loss_history": training_loss_history,
            "training_metric_history": training_metric_history,
            "validation_loss_history": validation_loss_history,
            "validation_metric_history": validation_metric_history,
        }

        # NOW DO THE ADJUSTMENTS USING THE LEARNING RATE SCHEDULER.
        if learning_rate_scheduler:
            learning_rate_scheduler(np.mean(validation_losses))
        # NOW DO THE ADJUSTMENTS USING THE EARLY STOPPING.
        if early_stopping:
            early_stopping(np.mean(validation_losses))
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
        "validation": {
            "loss": validation_loss_history,
            "metric": validation_metric_history,
        },
    }
    return result
