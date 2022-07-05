import logging
import typing
import torch
import pytorch_lightning as pl
import torch.nn as nn

logger = logging.getLogger(__name__)


class LitModel(pl.LightningModule):
    """Base class for Pytorch Lightning model - useful to reuse the same *_step methods."""
    def __init__(self):
        super().__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _generic_step(self, batch, batch_idx):
        """Runs the prediction + evaluation step for training/validation/testing."""
        inputs = batch['data']
        targets = batch['labels']
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets.float())
        return loss

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        """Runs a prediction step for testing, logging the loss."""
        loss = self._generic_step(batch, batch_idx)
        self.log("test_loss", loss)


class SimpleModel(LitModel):  # pragma: no cover
    """Simple Model Class.

    Inherits from the given framework's model class. This is a simple MLP model.
    """

    def __init__(
        self,
        num_classes,
    ):
        """__init__.

        Args:
            hyper_params (dict): hyper parameters from the config file.
        """
        super(SimpleModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5),
            nn.Conv2d(64, 64, 5),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.mlp_layers = nn.Sequential(
            nn.Linear(576, 128),  # The input size for the linear layer is determined by the previous operations
            nn.ReLU(),
            nn.Linear(128, num_classes),  # Here we get exactly num_classes logits at the output
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)  # Flatten is necessary to pass from CNNs to MLP
        x = self.mlp_layers(x)
        return x
