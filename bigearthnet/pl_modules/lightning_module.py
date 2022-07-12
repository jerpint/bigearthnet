import logging
import typing

import torch
import pytorch_lightning as pl
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

log = logging.getLogger(__name__)


class LitModule(pl.LightningModule):
    """Base class for Pytorch Lightning model - useful to reuse the same *_step methods."""
    def __init__(self, model):
        super().__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.model = model

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def _generic_step(self, batch, batch_idx):
        """Runs the prediction + evaluation step for training/validation/testing."""
        inputs = batch['data']
        targets = batch['labels']
        logits = self.model(inputs)
        loss = self.loss_fn(logits, targets.float())
        return {"loss": loss, "targets": targets, "logits": logits}

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        results = self._generic_step(batch, batch_idx)
        self.log("train_loss", results["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return results

    def _generic_epoch_end(self, step_outputs):
        class_names = self.trainer.train_dataloader.dataset.datasets.class_names

        all_targets = []
        all_preds = []
        for outputs in step_outputs:
            logits = outputs['logits']
            targets = outputs['targets']
            preds = (torch.sigmoid(logits) > 0.5)
            all_targets.extend(targets.numpy())
            all_preds.extend(preds.type(targets.dtype).numpy())

        conf_mats = multilabel_confusion_matrix(y_true=all_targets, y_pred=all_preds)
        report = classification_report(y_true=all_targets, y_pred=all_preds, target_names=class_names)
        return conf_mats, report

    def training_epoch_end(self, training_step_outputs):
        report, conf_mats = self._generic_epoch_end(training_step_outputs)
        log.info(f"Train epoch: {self.current_epoch}")
        log.info(f"Training Conf mats:\n{conf_mats}")
        log.info(f"Training classification report:\n{report}")
        # TODO: log this to tensorboard


    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
        results = self._generic_step(batch, batch_idx)
        self.log("val_loss", results["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return results

    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:
            report, conf_mats = self._generic_epoch_end(validation_step_outputs)
            log.info(f"Validation epoch: {self.current_epoch}")
            log.info(f"Validation Conf mats:\n{conf_mats}")
            log.info(f"Validation classification report:\n{report}")
            # TODO: log this to tensorboard

    def test_step(self, batch, batch_idx):
        """Runs a prediction step for testing, logging the loss."""
        results = self._generic_step(batch, batch_idx)
        self.log("test_loss", results["loss"])
