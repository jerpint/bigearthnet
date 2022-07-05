import pathlib
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

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

# initialize the model + datamodules
hub_dataset_path = pathlib.Path("./data/debug_dataset/hub_dataset/")
datamodule = load_datamodule(hub_path=hub_dataset_path, batch_size=16)
model = load_model(architecture="SimpleModel", num_classes=43)
callbacks = load_callbacks()

# do the training
trainer = pl.Trainer(max_epochs=5)  # gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
trainer = pl.Trainer(max_epochs=5, callbacks=callbacks)
trainer.fit(model, datamodule=datamodule)
