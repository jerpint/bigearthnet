import logging
import os
import socket

from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pip._internal.operations import freeze
from pytorch_lightning.callbacks import Callback

from bigearthnet.utils.reproducibility_utils import get_git_info

log = logging.getLogger(__name__)


class ReproducibilityLogging(Callback):
    """Log experiment details for reproducibility.

    This will pring the git hash, branch name, dependencies and
    omegaconf config to the log and save the omegaconf config to disk.
    """

    @staticmethod
    def parse_exp_details(cfg: DictConfig):  # pragma: no cover
        """Will parse the experiment details to a readable format for logging.

        :param cfg: (OmegaConf) The main config for the experiment.
        """
        # Log and save the config used for reproducibility
        script_path = get_original_cwd()
        git_hash, git_branch_name = get_git_info(script_path)
        hostname = socket.gethostname()
        dependencies = freeze.freeze()
        dependencies_str = "\n".join([d for d in dependencies])
        details = f"""
                  config: {OmegaConf.to_yaml(cfg)}
                  hostname: {hostname}
                  git code hash: {git_hash}
                  git branch name: {git_branch_name}
                  dependencies: {dependencies_str}
                  """
        return details

    def log_exp_info(self, trainer, pl_module):
        """Log info like the git branch, hash, dependencies, etc."""
        cfg = pl_module.cfg
        exp_details = self.parse_exp_details(cfg)
        log.info("Experiment info:" + exp_details + "\n")

        # dump the omegaconf config for reproducibility
        output_dir = os.path.join(trainer.logger.log_dir) if trainer.logger else "."
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        OmegaConf.save(cfg, os.path.join(output_dir, "exp_config.yaml"))

    def on_train_start(self, trainer, pl_module):
        self.log_exp_info(trainer, pl_module)
