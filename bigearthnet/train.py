import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from bigearthnet.pl_modules.lightning_module import LitModel
from bigearthnet.data.data_loader import load_datamodule

log = logging.getLogger(__name__)

def load_callbacks(cfg):
    all_callbacks = []

    last_model_checkpoint = callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="step",
        mode="max",
        filename="last-model",
    )
    all_callbacks.append(last_model_checkpoint)

    best_model_checkpoint = callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor=f"{cfg.monitor.name}/val",
        mode=cfg.monitor.mode,
        filename="best-model-{epoch:02d}-{step:02d}",
    )
    all_callbacks.append(best_model_checkpoint)

    early_stopping = callbacks.EarlyStopping(
        monitor=f"{cfg.monitor.name}/val",
        mode=cfg.monitor.mode,
        patience=cfg.monitor.patience,
    )
    all_callbacks.append(early_stopping)

    return all_callbacks


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
