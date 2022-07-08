import logging
import torch
from torch import optim


logger = logging.getLogger(__name__)


def load_optimizer(hyper_params, model):  # pragma: no cover
    """Instantiate the optimizer.

    Args:
        hyper_params (dict): hyper parameters from the config file
        model (obj): A neural network model object.

    Returns:
        optimizer (obj): The optimizer for the given model
    """
    optimizer_name = hyper_params['optimizer']
    # __TODO__ fix optimizer list
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters())
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters())
    else:
        raise ValueError('optimizer {} not supported'.format(optimizer_name))
    return optimizer
