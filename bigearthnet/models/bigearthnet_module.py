import importlib.resources
import json
import logging
import os
import typing

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.metrics import (
    classification_report,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
)
from torch import optim

from bigearthnet.utils.callbacks import _summarize_metrics

log = logging.getLogger(__name__)


def get_class_names():
    """Get the class names from the class_list.json file under bigearthnet/data/"""
    with importlib.resources.open_text("bigearthnet.data", "class_list.json") as file:
        return json.load(file)


class BigEarthNetModule(pl.LightningModule):
    """Base class for Pytorch Lightning model."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = instantiate(cfg.model)
        self.save_hyperparameters(cfg, logger=False)
        self.init_loss()

        self.class_names = get_class_names()

    def init_loss(self):
        weights_file = self.cfg.loss.get("class_weights")
        if weights_file:
            # If specified in the config, the loss will be rebalanced according to data.
            assert os.path.isfile(weights_file)
            log.info(f"loading {weights_file} as weights for the loss function")
            with open(weights_file, "r") as f:
                data = json.load(f)
            pos_weight = torch.tensor(list(data.values()))
        else:
            pos_weight = None
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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
            y_true=all_targets,
            y_pred=all_preds,
            average="micro",
            zero_division=0,
        )
        avg_loss = sum(all_loss) / len(all_loss)
        conf_mats = multilabel_confusion_matrix(
            y_true=all_targets, y_pred=all_preds, labels=range(len(self.class_names))
        )
        report = classification_report(
            y_true=all_targets,
            y_pred=all_preds,
            target_names=self.class_names,
            zero_division=0,
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
            self.val_metrics = metrics  # cache for use in callback
            self.log_metrics(metrics, split="val")

    def test_step(self, batch, batch_idx):
        """Runs a predictionval_ step for testing, logging the loss."""
        outputs = self._generic_step(batch, batch_idx)
        return outputs

    def test_epoch_end(self, test_step_outputs):
        metrics = self._generic_epoch_end(test_step_outputs)
        self.test_metrics = metrics
        self.log_metrics(metrics, split="test")

    def log_metrics(self, metrics: typing.Dict, split: str):
        """Logs all metrics to logs and to tensorboard."""
        assert split in ["train", "val", "test"]

        # log our metrics to the logs directly
        metrics_summary = _summarize_metrics(
            metrics,
            self.class_names,
            split,
            self.current_epoch,
        )
        log.info(metrics_summary)

        # log metrics to tensorboard
        if split in ["train", "val"]:
            self.log(f"precision/{split}", metrics["precision"], on_epoch=True)
            self.log(f"recall/{split}", metrics["recall"], on_epoch=True)
            self.log(f"f1_score/{split}", metrics["f1_score"], on_epoch=True)
