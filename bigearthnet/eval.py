import logging
import os
import pathlib

import hydra
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from bigearthnet.models.bigearthnet_module import BigEarthNetModule
from bigearthnet.utils.reproducibility_utils import set_seed

log = logging.getLogger(__name__)


@hydra.main(config_name="exp_config", version_base="1.2")
def main(cfg: DictConfig):

    log.info("Evaluating model...")

    # instantiate all objects from hydra configs
    model = BigEarthNetModule(cfg)
    datamodule = instantiate(cfg.datamodule)
    trainer = instantiate(cfg.trainer)
    datamodule.setup()

    # Here, cfg_path is the path to the hydra config of the experiment that was run
    cfg_path = HydraConfig.get().get("runtime").get("config_sources")[1]["path"]
    os.chdir(cfg_path)

    # Retrieve the best model (should be under checkpoints/best_model.cpkt)
    ckpt_path = str(list(pathlib.Path(cfg_path).rglob("best-model*.ckpt"))[0].resolve())

    # Evaluate best model on test set
    trainer.test(model=model, ckpt_path=ckpt_path, datamodule=datamodule)
    log.info("Test evaluation Done.")


if __name__ == "__main__":
    main()
