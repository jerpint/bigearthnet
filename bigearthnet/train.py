import pathlib

import torch
import pytorch_lightning as pl

from bigearthnet.data.data_loader import DataModule
from bigearthnet.models.my_model import SimpleModel



# load datamodules
hub_dataset_path = pathlib.Path("./data/debug_dataset/hub_dataset/")
dm = DataModule(dataset_path=hub_dataset_path, batch_size=16)
dm.setup()


# initialize the model
model = SimpleModel(num_classes=43)

# do the training
trainer = pl.Trainer(max_epochs=5) # gpus=4, num_nodes=8, precision=16, limit_train_batches=0.5)
trainer.fit(model, datamodule=dm)
