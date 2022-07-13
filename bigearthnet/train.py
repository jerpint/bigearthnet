import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from bigearthnet.pl_modules.lightning_module import LitModel
from bigearthnet.data.data_loader import load_datamodule

log = logging.getLogger(__name__)

def load_callbacks(cfg):
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
        monitor=f"{cfg.monitor.name}/val",
        mode=cfg.monitor.mode,
        filename="best-model-{epoch:02d}-{val_loss:.2f}",
    )
    callbacks.append(best_model_checkpoint)

    return callbacks


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):

    log.info("Beginning training...")

    # Log and save the config used for reproducibility
    log.info(f"Configurations specified: {cfg}")
    OmegaConf.save(cfg, "exp_config.yaml")

    # data setup
    model = LitModel(cfg)
    datamodule = load_datamodule(cfg)
    callbacks = load_callbacks(cfg)
    logger = TensorBoardLogger(**cfg.logger)

    # do the training
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model, datamodule=datamodule)

    log.info("Training Done.")

if __name__ == "__main__":
    main()
