import logging
import typing

import torch
from torch import optim
import pytorch_lightning as pl
from hydra.utils import instantiate
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support

log = logging.getLogger(__name__)


class LitModel(pl.LightningModule):
    """Base class for Pytorch Lightning model - useful to reuse the same *_step methods."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = instantiate(cfg.model)
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    def configure_optimizers(self):

        name = self.cfg.optimizer.name
        lr = self.cfg.optimizer.lr
        if name == 'adam':
            #  optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
            optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=lr,
            )
        elif name == 'sgd':
            optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=lr
            )
        else:
            raise ValueError(f'optimizer {name} not supported')
        return optimizer

    def _generic_step(self, batch, batch_idx):
        """Runs the prediction + evaluation step for training/validation/testing."""
        inputs = batch['data']
        targets = batch['labels']
        logits = self.model(inputs)
        loss = self.loss_fn(logits, targets.float())
        return {"loss": loss, "targets": targets, "logits": logits}

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

        prec, rec, f1, s = precision_recall_fscore_support(y_true=all_targets, y_pred=all_preds, average="micro")
        conf_mats = multilabel_confusion_matrix(y_true=all_targets, y_pred=all_preds)
        report = classification_report(y_true=all_targets, y_pred=all_preds, target_names=class_names)

        metrics = {
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'conf_mats': conf_mats,
                'report': report,
        }
        return metrics

    def training_step(self, batch, batch_idx):
        """Runs a prediction step for training, returning the loss."""
        outputs = self._generic_step(batch, batch_idx)
        self.log("loss/train", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return outputs

    def training_epoch_end(self, training_step_outputs):
        metrics = self._generic_epoch_end(training_step_outputs)
        self.log_metrics(metrics, split="train")
        # TODO: log this to tensorboard

    def validation_step(self, batch, batch_idx):
        """Runs a prediction step for validation, logging the loss."""
        outputs = self._generic_step(batch, batch_idx)
        self.log("loss/val", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return outputs

    def log_metrics(self, metrics: typing.Dict, split: str):
        # log to tensorboard
        self.log(f"precision/{split}", metrics["precision"], on_epoch=True)
        self.log(f"recall/{split}", metrics["recall"], on_epoch=True)
        self.log(f"f1_score/{split}", metrics["f1_score"], on_epoch=True)

        # add to logs
        log.info(f"{split} epoch: {self.current_epoch}")
        log.info(f"{split} Conf mats:\n{metrics['conf_mats']}")
        log.info(f"{split} classification report:\n{metrics['report']}")


    def validation_epoch_end(self, validation_step_outputs):
        if not self.trainer.sanity_checking:
            metrics = self._generic_epoch_end(validation_step_outputs)
            self.log_metrics(metrics, split="val")

    def test_step(self, batch, batch_idx):
        """Runs a prediction step for testing, logging the loss."""
        outputs = self._generic_step(batch, batch_idx)
        self.log("test_loss", outputs["loss"])
