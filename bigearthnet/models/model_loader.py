import logging

from bigearthnet.pl_modules.lightning_module import LitModule


logger = logging.getLogger(__name__)


def load_model(architecture, num_classes):  # pragma: no cover
    """Instantiate a model.

    Args:
        hyper_params (dict): hyper parameters from the config file

    Returns:
        model (obj): A neural network model object.
    """
    if architecture == 'baseline':
        from bigearthnet.models.my_model import Baseline
        model = Baseline(num_classes=num_classes)
    elif architecture == 'resnet50d':
        import timm
        model = timm.create_model(architecture, pretrained=False, num_classes=num_classes)
    else:
        raise ValueError('architecture {} not supported'.format(architecture))
    logger.info('selected architecture: {}'.format(architecture))
    logger.info('model info:\n' + str(model) + '\n')

    return LitModule(model)
