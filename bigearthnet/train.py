
import datetime
import glob
import logging
import os
import shutil

import mlflow
import orion
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from orion.client import report_results
from yaml import dump
from yaml import load

from bigearthnet.utils.hp_utils import check_and_log_hp

logger = logging.getLogger(__name__)

BEST_MODEL_NAME = 'best_model'
LAST_MODEL_NAME = 'last_model'
STAT_FILE_NAME = 'stats.yaml'


def train(**kwargs):  # pragma: no cover
    """Training loop wrapper. Used to catch exception if Orion is being used."""
    try:
        best_dev_metric = train_impl(**kwargs)
    except RuntimeError as err:
        if orion.client.cli.IS_ORION_ON and 'CUDA out of memory' in str(err):
            logger.error(err)
            logger.error('model was out of memory - assigning a bad score to tell Orion to avoid'
                         'too big model')
            best_dev_metric = -999
        else:
            raise err

    report_results([dict(
        name='dev_metric',
        type='objective',
        # note the minus - cause orion is always trying to minimize (cit. from the guide)
        value=-float(best_dev_metric))])


def load_mlflow(output):
    """Load the mlflow run id.

    Args:
        output (str): Output directory
    """
    with open(os.path.join(output, STAT_FILE_NAME), 'r') as stream:
        stats = load(stream, Loader=yaml.FullLoader)
    return stats['mlflow_run_id']


def write_mlflow(output):
    """Write the mlflow info to resume the training plotting..

    Args:
        output (str): Output directory
    """
    mlflow_run = mlflow.active_run()
    mlflow_run_id = mlflow_run.info.run_id if mlflow_run is not None else 'NO_MLFLOW'
    to_store = {'mlflow_run_id': mlflow_run_id}
    with open(os.path.join(output, STAT_FILE_NAME), 'w') as stream:
        dump(to_store, stream)


def train_impl(model, datamodule, output, hyper_params,
               use_progress_bar, start_from_scratch, mlf_logger, gpus):  # pragma: no cover
    """Main training loop implementation.

    Args:
        model (obj): The neural network model object.
        datamodule (obj): lightning data module that will instantiate data loaders.
        output (str): Output directory.
        hyper_params (dict): Dict containing hyper-parameters.
        use_progress_bar (bool): Use tqdm progress bar (can be disabled when logging).
        start_from_scratch (bool): Start training from scratch (ignore checkpoints)
        mlf_logger (obj): MLFlow logger callback.
        gpus: number of GPUs to use.
    """
    check_and_log_hp(['max_epoch', 'patience'], hyper_params)
    write_mlflow(output)

    best_model_path = os.path.join(output, BEST_MODEL_NAME)
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=best_model_path,
        filename='model',
        save_top_k=1,
        verbose=use_progress_bar,
        monitor="val_loss",
        mode="auto",
        period=1,
    )

    last_model_path = os.path.join(output, LAST_MODEL_NAME)
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=last_model_path,
        filename='model',
        verbose=use_progress_bar,
        period=1,
    )

    resume_from_checkpoint = handle_previous_models(output, last_model_path, best_model_path,
                                                    start_from_scratch)

    early_stopping = EarlyStopping("val_loss", mode="auto", patience=hyper_params['patience'],
                                   verbose=use_progress_bar)
    trainer = pl.Trainer(
        callbacks=[early_stopping, best_checkpoint_callback, last_checkpoint_callback],
        checkpoint_callback=True,
        logger=mlf_logger,
        max_epochs=hyper_params['max_epoch'],
        resume_from_checkpoint=resume_from_checkpoint,
        gpus=gpus
    )

    trainer.fit(model, datamodule=datamodule)
    best_dev_result = float(early_stopping.best_score.cpu().numpy())
    return best_dev_result


def handle_previous_models(output, last_model_path, best_model_path, start_from_scratch):
    """Moves the previous models in a new timestamp folder."""
    last_models = glob.glob(last_model_path + os.sep + '*')
    best_models = glob.glob(best_model_path + os.sep + '*')

    if len(last_models + best_models) > 0:
        now = datetime.datetime.now()
        timestamp = now.strftime('%Y%m%d_%H%M%S')
        new_folder = output + os.path.sep + timestamp
        os.mkdir(new_folder)
        shutil.move(last_model_path, new_folder)
        shutil.move(best_model_path, new_folder)
        logger.info(f'old models found - moving them to {new_folder}')
        # need to change the last model pointer to the new location
        last_models = glob.glob(new_folder + os.path.sep + LAST_MODEL_NAME + os.sep + '*')

    if start_from_scratch:
        logger.info('will not load any pre-existent checkpoint (because of "--start-from-scratch")')
        resume_from_checkpoint = None
    elif len(last_models) >= 1:
        resume_from_checkpoint = sorted(last_models)[-1]
        logger.info(f'models found - resuming from {resume_from_checkpoint}')
    else:
        logger.info('no model found - starting training from scratch')
        resume_from_checkpoint = None
    return resume_from_checkpoint
