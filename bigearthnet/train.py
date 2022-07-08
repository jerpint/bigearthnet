import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import hydra
from omegaconf import DictConfig

from bigearthnet.data.data_loader import load_datamodule
from bigearthnet.models.model_loader import load_model

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

    # data setup
    datamodule = load_datamodule(**cfg.datamodule)
    model = load_model(**cfg.model)
    callbacks = load_callbacks()

    # do the training
    trainer = pl.Trainer(**cfg.trainer, callbacks=callbacks)
    trainer.fit(model, datamodule=datamodule)

if __name__ == "__main__":
    main()
