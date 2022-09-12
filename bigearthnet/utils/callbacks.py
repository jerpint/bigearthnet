import logging
import os
import socket
import typing

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


class MonitorHyperParameters(Callback):
    """Keeps track of hyper parameters in tensorboard.

    Useful for generating parallel coordinates view.
    """

    @staticmethod
    def extract_hparams(cfg) -> typing.Dict:
        """Select which of the config params to log in logger."""
        hparams = {
            "optimizer": cfg.optimizer,
            "transforms": cfg.transforms.description,
            "datamodule": {
                k: cfg.datamodule[k] for k in ["batch_size", "dataset_name"]
            },
            "model": cfg.model,
        }
        if cfg.model.get("pretrained"):
            # tensorboard doesn't log bool values, convert to int
            hparams["model"]["pretrained"] = int(hparams["model"]["pretrained"])
        return hparams

    def init_hparams_metrics(self, trainer, pl_module):
        # Set up initial metrics associated to hparams before training
        init_metrics = {
            "val_best_metrics/loss": 99999,
            "val_best_metrics/precision": 0,
            "val_best_metrics/recall": 0,
            "val_best_metrics/f1_score": 0,
        }

        # keep track of the best value in the pl_module
        monitor_name = pl_module.cfg.monitor.name
        pl_module.val_best_metric = init_metrics[f"val_best_metrics/{monitor_name}"]

        # Log the initialized values to tensorboard
        trainer.logger.log_hyperparams(
            self.extract_hparams(pl_module.cfg), metrics=init_metrics
        )

    @staticmethod
    def requires_update(metrics, mode, name, best_value):
        if mode == "min" and metrics[name] < best_value:
            return True
        if mode == "max" and metrics[name] > best_value:
            return True
        return False

    def update_best_metric(self, trainer, pl_module):
        """Update the best scoring metric for parallel coordinate plots."""
        val_metrics = pl_module.val_metrics
        mode = pl_module.cfg.monitor.mode
        name = pl_module.cfg.monitor.name
        best_value = pl_module.val_best_metric

        if self.requires_update(val_metrics, mode, name, best_value):
            trainer.logger.log_hyperparams(
                self.extract_hparams(pl_module.cfg),
                metrics={
                    f"val_best_metrics/{k}": val_metrics[k]
                    for k in ["loss", "precision", "recall", "f1_score"]
                },
            )
            pl_module.val_best_metric = val_metrics[name]

    def on_train_start(self, trainer, pl_module):
        self.init_hparams_metrics(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self.update_best_metric(trainer, pl_module)
