import logging
import os
import socket
import typing

import matplotlib.pyplot as plt
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pip._internal.operations import freeze
from pytorch_lightning.callbacks import Callback
from sklearn.metrics import ConfusionMatrixDisplay

from bigearthnet.utils.reproducibility_utils import get_git_info

log = logging.getLogger(__name__)


def _plot_conf_mats(
    conf_mats: typing.List, class_names: typing.List[str], title: str = ""
):
    """Creates a matplotlib figure with each subplot a unique confusion matrix."""
    conf_mat_fig, axs = plt.subplots(9, 5, figsize=(12, 15))
    [ax.set_axis_off() for ax in axs.ravel()]  # turn all axes off
    for cm, label, ax in zip(conf_mats, class_names, axs.ravel()):

        # add to figure
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
        )
        disp.plot(ax=ax, colorbar=False)
        ax.title.set_text(label[0:20])  # text cutoff for displaying

    conf_mat_fig.suptitle(title)

    return conf_mat_fig


def _log_conf_mats(conf_mats: typing.List, class_names: typing.List[str]):
    """Parse confusion matrices to plaintext with class names."""
    conf_mat_log = ""
    for cm, label in zip(conf_mats, class_names):
        conf_mat_log += f"\n{label}\n{cm}\n"
    return conf_mat_log


def _summarize_metrics(metrics, class_names, split, current_epoch=None):
    """Summarize all metrics to a blob of plaintext."""
    classification_report = metrics["report"]
    conf_mats = metrics["conf_mats"]
    conf_mat_log = _log_conf_mats(conf_mats, class_names)

    return f"""
    {split} Epoch: {current_epoch}
    {split} Classification report:\n{classification_report}
    {split} Confusion matrices:\n{conf_mat_log}
    """


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
            "val_best_metrics/loss": float("inf"),
            "val_best_metrics/precision": float("-inf"),
            "val_best_metrics/recall": float("-inf"),
            "val_best_metrics/f1_score": float("-inf"),
        }

        # verify that the value we want to monitor is valid
        monitor_name = pl_module.cfg.monitor.name
        possible_monitor_names = ["loss", "precision", "recall", "f1_score"]
        if monitor_name not in possible_monitor_names:
            raise ValueError(
                f"Specified monitor.name as {monitor_name}. Value to monitor must be one of {possible_monitor_names}"
            )

        # set the best value as the initial value
        pl_module.val_best_metric = init_metrics[f"val_best_metrics/{monitor_name}"]

        # Log the initialized values to tensorboard
        trainer.logger.log_hyperparams(
            self.extract_hparams(pl_module.cfg), metrics=init_metrics
        )

    @staticmethod
    def requires_update(metrics, mode, name, best_value):
        assert mode in ["min", "max"]
        current_value = metrics[name]
        if mode == "min" and current_value < best_value:
            return True
        if mode == "max" and current_value > best_value:
            return True
        return False

    def save_best_metrics(self, trainer, pl_module):
        class_names = pl_module.class_names
        current_epoch = pl_module.current_epoch
        metrics = pl_module.val_metrics

        metrics_summary = _summarize_metrics(
            metrics=metrics,
            class_names=class_names,
            split="val",
            current_epoch=current_epoch,
        )
        output_dir = os.path.join(trainer.logger.log_dir) if trainer.logger else "."
        with open(os.path.join(output_dir, "val_best_metrics.txt"), "w") as f:
            f.write(metrics_summary)

        # Generate the figure with confusion matrices
        # and plots it to tensorboard
        conf_mats = metrics["conf_mats"]
        fig_title = f"f1 score: {metrics['f1_score']:2.2f}\nepoch: {current_epoch}"
        conf_mat_figure = _plot_conf_mats(conf_mats, class_names, title=fig_title)
        trainer.logger.experiment.add_figure(
            tag=f"best_confusion_matrix/val",
            figure=conf_mat_figure,
            global_step=pl_module.global_step,
        )
        plt.close(conf_mat_figure)

    def update_best_metric(self, trainer, pl_module):
        """Update the best scoring metric for parallel coordinate plots

        Saves a copy of the best metrics to disk for later use.
        """
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

            # save to disk
            self.save_best_metrics(trainer, pl_module)

    def on_train_start(self, trainer, pl_module):
        self.init_hparams_metrics(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self.update_best_metric(trainer, pl_module)
