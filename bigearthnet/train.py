import logging

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from bigearthnet.data.data_loader import load_datamodule
from bigearthnet.models.model_loader import load_model

log = logging.getLogger(__name__)

def load_callbacks():
    callbacks = []

    last_model_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="step",
        mode="max",
        filename="last-model",
    )
    callbacks.append(last_model_checkpoint)

    best_model_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor="val_loss",
        mode='min',
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
    )
    callbacks.append(best_model_checkpoint)

    return callbacks


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):

    log.info("Beginning training...")
    log.info(f"Configurations specified: {cfg}")

    # data setup
    datamodule = load_datamodule(**cfg.datamodule)
    model = load_model(**cfg.model)
    callbacks = load_callbacks()
    logger = TensorBoardLogger(**cfg.logger)

    # do the training
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=datamodule)

    log.info("Training Done.")

if __name__ == "__main__":
    main()
