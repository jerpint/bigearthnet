import logging

from bigearthnet.models.my_model import SimpleModel

logger = logging.getLogger(__name__)


def load_model(architecture, num_classes):  # pragma: no cover
    """Instantiate a model.

    Args:
        hyper_params (dict): hyper parameters from the config file

    Returns:
        model (obj): A neural network model object.
    """
    if architecture == 'SimpleModel':
        model = SimpleModel(num_classes=num_classes)
    else:
        raise ValueError('architecture {} not supported'.format(architecture))
    logger.info('selected architecture: {}'.format(architecture))
    logger.info('model info:\n' + str(model) + '\n')


    return model
