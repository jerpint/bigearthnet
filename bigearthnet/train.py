import logging
import pathlib

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from bigearthnet.pl_modules.lightning_module import LitModel

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):

    log.info("Beginning training...")

    # Log and save the config used for reproducibility
    log.info(f"Configurations specified: {cfg}")
    OmegaConf.save(cfg, "exp_config.yaml")

    # instantiate all objects from hydra configs
    model = LitModel(cfg)
    datamodule = instantiate(cfg.datamodule)
    callbacks = instantiate(cfg.callbacks)
    logger = instantiate(cfg.logger)
    trainer = instantiate(cfg.trainer)

    # do the training
    datamodule.setup()
    trainer.fit(model, datamodule=datamodule)
    log.info("Training Done.")

    ckpt_path = str(list(pathlib.Path(".").rglob("best-model*.ckpt"))[0].resolve())
    trainer.test(ckpt_path=ckpt_path, datamodule=datamodule)
    log.info("Test evaluation Done.")


if __name__ == "__main__":
    main()
