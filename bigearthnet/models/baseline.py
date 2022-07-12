import logging
import typing

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class Baseline(torch.nn.Module):  # pragma: no cover
    """Baseline Model Class.

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
        super(Baseline, self).__init__()
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
