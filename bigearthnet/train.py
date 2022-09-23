import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from bigearthnet.models.bigearthnet_module import BigEarthNetModule
from bigearthnet.utils.reproducibility_utils import set_seed

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="config", version_base="1.2")
def main(cfg: DictConfig):

    log.info("Beginning training...")

    # set seed if specified
    if cfg.experiment.get("seed"):
        set_seed(cfg.experiment.seed)

    # instantiate all objects from hydra configs
    model = BigEarthNetModule(cfg)
    datamodule = instantiate(cfg.datamodule)
    datamodule.setup()

    trainer = instantiate(cfg.trainer)

    # do the training
    trainer.fit(model, datamodule=datamodule)
    log.info("Training Done.")


if __name__ == "__main__":
    main()
