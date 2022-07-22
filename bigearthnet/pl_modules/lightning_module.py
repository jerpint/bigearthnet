import logging
import typing
import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from torch import optim

from bigearthnet.utils.reproducibility_utils import get_exp_details

log = logging.getLogger(__name__)


class LitModel(pl.LightningModule):
    """Base class for Pytorch Lightning model."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.model)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

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

    def log_exp_info(self):
        """Log info like the git branch, hash, dependencies, etc."""
        exp_details = get_exp_details(self.cfg)
        log.info("Experiment info:" + exp_details + "\n")
        self.logger.experiment.add_text("exp_details", exp_details)

        # dump the config for reproducibility
        output_dir = os.path.join(self.logger.log_dir) if self.logger else "."
        OmegaConf.save(self.cfg, os.path.join(output_dir, "exp_config.yaml"))

    def init_hparams(self):
        mode = self.cfg.monitor.mode
        name = self.cfg.monitor.name
        assert mode in ["min", "max"]
        assert name in ["loss", "precision", "recall", "f1_score"]
        # initial metrics before training
        init_metrics = {
            "val_best_metrics/loss": 99999,
            "val_best_metrics/precision": 0,
            "val_best_metrics/recall": 0,
            "val_best_metrics/f1_score": 0,
        }

        self.logger.log_hyperparams(
            self.extract_hparams(self.cfg), metrics=init_metrics
        )

        self.val_best_metric = init_metrics[f"val_best_metrics/{name}"]

    def on_train_start(self):
        # log experiment details for reproducibility
        self.log_exp_info()

        self.init_hparams()

        # get the class names (for later use)
        self.class_names: typing.List = (
            self.trainer.train_dataloader.dataset.datasets.class_names
        )

    def configure_optimizers(self):
        name = self.cfg.optimizer.name
        lr = self.cfg.optimizer.lr
        if name == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
            )
        elif name == "sgd":
            optimizer = optim.SGD(self.model.parameters(), lr=lr)
        else:
            raise ValueError(f"optimizer {name} not supported")
        return optimizer

    def _generic_step(self, batch, batch_idx):
        """Runs the prediction + evaluation step for training/validation/testing."""
        inputs = batch["data"]
        targets = batch["labels"]
        logits = self.model(inputs)
        loss = self.loss_fn(logits, targets.float())
        return {"loss": loss, "targets": targets, "logits": logits}

    def _generic_epoch_end(self, step_outputs):

        all_targets = []
        all_preds = []
        all_loss = []
        for outputs in step_outputs:
            logits = outputs["logits"]
            targets = outputs["targets"]
            preds = torch.sigmoid(logits) > 0.5
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.type(targets.dtype).cpu().numpy())

            loss = outputs["loss"]
            all_loss.append(loss.cpu().numpy())

        prec, rec, f1, s = precision_recall_fscore_support(
            y_true=all_targets, y_pred=all_preds, average="micro"
        )
        avg_loss = sum(all_loss) / len(all_loss)
        conf_mats = multilabel_confusion_matrix(y_true=all_targets, y_pred=all_preds)
        report = classification_report(
            y_true=all_targets, y_pred=all_preds, target_names=self.class_names
        )

        metrics = {
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "conf_mats": conf_mats,
            "report": report,
            "loss": avg_loss,
        }
        return metrics

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        outputs = self._generic_step(batch, batch_idx)
        self.log(
            "loss/train",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs

    def training_epoch_end(self, training_step_outputs):
        metrics = self._generic_epoch_end(training_step_outputs)
        self.log_metrics(metrics, split="train")

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
        outputs = self._generic_step(batch, batch_idx)
        self.log(
            "loss/val",
            outputs["loss"],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return outputs

    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:
            metrics = self._generic_epoch_end(validation_step_outputs)
            self.log_metrics(metrics, split="val")
            self.update_best_metric(metrics)

    def test_step(self, batch, batch_idx):
        """Runs a predictionval_ step for testing, logging the loss."""
        outputs = self._generic_step(batch, batch_idx)
        return outputs

    def test_epoch_end(self, test_step_outputs):
        metrics = self._generic_epoch_end(test_step_outputs)
        self.log_metrics(metrics, split="test")

    def log_metrics(self, metrics: typing.Dict, split: str):
        assert split in ["train", "val", "test"]
        # print to logs
        log.info(f"{split} epoch: {self.current_epoch}")
        log.info(f"{split} classification report:\n{metrics['report']}")

        # Here we prepare logs for the confusion matrices (plots and text summary)
        conf_mats = metrics["conf_mats"]
        conf_mat_log = f"{split} Confusion matrices:\n:"
        conf_mat_fig, axs = plt.subplots(9, 5, figsize=(12, 15))
        [ax.set_axis_off() for ax in axs.ravel()]  # default turn all axes off
        for cm, label, ax in zip(conf_mats, self.class_names, axs.ravel()):
            # add to log
            conf_mat_log += f"\n{label}\n{cm}\n"

            # add to figure
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
            )
            disp.plot(ax=ax, colorbar=False)
            ax.title.set_text(label[0:20])  # text cutoff

        # log to tensorboard
        if split in ["train", "val"]:
            self.log(f"precision/{split}", metrics["precision"], on_epoch=True)
            self.log(f"recall/{split}", metrics["recall"], on_epoch=True)
            self.log(f"f1_score/{split}", metrics["f1_score"], on_epoch=True)
            self.logger.experiment.add_figure(
                f"confusion matrix/{split}", conf_mat_fig, self.global_step
            )
        log.info(conf_mat_log)
        plt.close(conf_mat_fig)

    def update_best_metric(self, metrics):
        """Update the best scoring metric for parallel coordinate plots."""
        mode = self.cfg.monitor.mode
        name = self.cfg.monitor.name
        update = False
        if mode == "min" and metrics[name] < self.val_best_metric:
            update = True
        if mode == "max" and metrics[name] > self.val_best_metric:
            update = True
        if update:
            self.logger.log_hyperparams(
                self.extract_hparams(self.cfg),
                metrics={
                    f"val_best_metrics/{k}": metrics[k]
                    for k in ["loss", "precision", "recall", "f1_score"]
                },
            )
            self.val_best_metric = metrics[name]

            output_dir = os.path.join(self.logger.log_dir) if self.logger else "."
            np.save(os.path.join(output_dir, "val_best_metrics.npy"), metrics)
