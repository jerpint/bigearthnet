import logging
import typing
import os

from pytorch_lightning.callbacks import Callback
from omegaconf import DictConfig, OmegaConf

from bigearthnet.utils.reproducibility_utils import get_exp_details

log = logging.getLogger(__name__)


class ReproducibilityLogging(Callback):
    """Log experiment details for reproducibility.

    This will pring the git hash, branch name, dependencies and
    omegaconf config to the log and save the omegaconf config to disk.
    """

    @staticmethod
    def log_exp_info(trainer, pl_module):
        """Log info like the git branch, hash, dependencies, etc."""
        cfg = pl_module.cfg
        exp_details = get_exp_details(cfg)
        log.info("Experiment info:" + exp_details + "\n")

        # dump the omegaconf config for reproducibility
        output_dir = os.path.join(trainer.logger.log_dir) if trainer.logger else "."
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        OmegaConf.save(cfg, os.path.join(output_dir, "exp_config.yaml"))

    def on_train_start(self, trainer, pl_module):
        self.log_exp_info(trainer, pl_module)
