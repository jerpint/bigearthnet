import argparse
import logging

import pytorch_lightning as pl
from hydra.utils import instantiate

from bigearthnet.datamodules.bigearthnet_datamodule import BigEarthNetDataModule
from bigearthnet.models.bigearthnet_module import BigEarthNetModule
from bigearthnet.utils.callbacks import MonitorHyperParameters

logger = logging.getLogger(__name__)


def main(
    ckpt_path, dataset_dir, dataset_name, batch_size, num_workers, accelerator, devices
):

    logger.info("Evaluating model...")

    # Load the model from the checkpoint
    model = BigEarthNetModule.load_from_checkpoint(ckpt_path)

    # fetch the transforms used in the model
    transforms = instantiate(model.cfg.transforms.obj)

    # instantiate the datamodule
    datamodule = BigEarthNetDataModule(
        dataset_dir, dataset_name, batch_size, num_workers, transforms
    )
    datamodule.setup()

    # This callback will save all the best metrics to files
    callbacks = [MonitorHyperParameters()]
    trainer = pl.Trainer(callbacks=callbacks, accelerator=accelerator, devices=devices)

    # Evaluate best model on test set
    trainer.test(model=model, datamodule=datamodule)
    logger.info("Test evaluation Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-path",
        help="Path to best model's checkpoint.",
    )
    parser.add_argument(
        "--dataset-dir",
        help="""
        Path to where datasets are stored.
        Assumes to be ../datasets by default, needs to be updated if run outside of top-level directory.
        """,
        default="../datasets/",
    )
    parser.add_argument(
        "--dataset-name",
        help="""
        Name of the dataset to use for evaluation.
        Will load that dataset and evaluate on its test set.
        Must be one of:
            ["bigearthnet-mini", "bigearthnet-medium", bigearthnet-full"].
        """,
        default="bigearthnet-mini",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
    )
    parser.add_argument(
        "--num-workers",
        default=0,
    )
    parser.add_argument(
        "--accelerator",
        help="One of ['cpu', 'gpu']. Uses 'cpu' by default.",
        default="cpu",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="Specify how many gpus to use when avaialble (suggested to use 1)",
    )
    args = parser.parse_args()

    main(
        args.ckpt_path,
        args.dataset_dir,
        args.dataset_name,
        args.batch_size,
        args.num_workers,
        args.accelerator,
        args.devices,
    )
