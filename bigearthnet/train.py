import logging
import pathlib

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from bigearthnet.pl_modules.lightning_module import LitModel
from bigearthnet.utils.reproducibility_utils import set_seed

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):

    log.info("Beginning training...")

    # set seed if explicitly passed
    if cfg.experiment.get("seed"):
        log.info(f"Setting seed to: {cfg.experiment.seed}")
        set_seed(cfg.experiment.seed)

    # instantiate all objects from hydra configs
    model = LitModel(cfg)
    datamodule = instantiate(cfg.datamodule)
    trainer = instantiate(cfg.trainer)

    # do the training
    datamodule.setup()
    trainer.fit(model, datamodule=datamodule)
    log.info("Training Done.")

    # Evaluate best model on test set
    ckpt_path = str(list(pathlib.Path(".").rglob("best-model*.ckpt"))[0].resolve())
    trainer.test(ckpt_path=ckpt_path, datamodule=datamodule)
    log.info("Test evaluation Done.")


if __name__ == "__main__":
    main()
