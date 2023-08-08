"""
Code copied from:
https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/
"""
import torch


# TODO: Maybe implement more learning rate scheduler options.
class LRScheduler:
    """
    Learning rate scheduler. If the validation loss does not decrease for the
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """

    def __init__(self, optimizer, patience: int = 5, min_lr: float = 1e-6, factor=0.5):
        """new_lr = old_lr * factor

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            The optimizer we are using.
        patience : int, optional
            How many epochs to wait before updating the learning rate, by default 5
        min_lr : float, optional
            Least learning rate value to reduct to while updating, by default 1e-6
        factor : float, optional
            Factor by which the learning rate should be updated, by default 0.5
        """
        self._optimizer = optimizer
        self._patience = patience
        self._min_lr = min_lr
        self._factor = factor

        self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            patience=self._patience,
            factor=self._factor,
            min_lr=self._min_lr,
            verbose=True,
        )

    def __call__(self, val_loss):
        self._lr_scheduler.step(val_loss)
