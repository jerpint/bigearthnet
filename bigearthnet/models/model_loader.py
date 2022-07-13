import logging

from hydra.utils import instantiate
from bigearthnet.pl_modules.lightning_module import LitModule


logger = logging.getLogger(__name__)


def load_model(cfg):  # pragma: no cover
    """Instantiate a model.

    Args:
        hyper_params (dict): hyper parameters from the config file

    Returns:
        model (obj): A neural network model object.
    """
    model = instantiate(cfg.model)
    logger.info('model info:\n' + str(model) + '\n')

    return LitModule(model)
