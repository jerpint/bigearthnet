# BigEarthNet 🌎

Welcome to the  [bigearthnet](https://bigearth.net/) classification repo!  This project was built as an example project built using modern tools for applied deep leaerning in the context of an [applied deep learning course](https://catalogue.ivado.umontreal.ca/Web/MyCatalog/ViewP?id=bfBsk1MpdKCuFXDwCrvZMw%3d%3d&pid=DwpGfXsYFQ5dNLAWEt9mWQ%3d%3d&mc_cid=c7c86767f2&mc_eid=3258909e75)

Here are a few of the features baked-in to this repo:

* Pytorch-Lightning: Implements all the training loops and boilerplate code
* [Hydra](hydra.cc): Easily manage and configure experiment parameters
* [TIMM](https://github.com/rwightman/pytorch-image-models): A model zoo of SOTA pre-trained classification models
* [Tensorboard](https://www.tensorflow.org/tensorboard): To track experiment progress
* [Deep Lake / Activeloop Hub](https://github.com/activeloopai/deeplake): An efficient dataset generator.

# Getting Started

Here are the basic steps for getting setup on most mochines:

## 1. Clone this project

    git clone https://github.com/jerpint/bigearthnet

## 2. Install the dependencies:
It is recommended to work in a virtual environment (e.g. conda):

    conda create -n bigearth python=3.8
    conda activate bigearth

Once activated, run:

    cd ~/bigearthnet/
    pip install -e .


## 3. Test your installation
To test your install, simply run:

    cd ~/bigearthnet/bigearthnet
    python train.py

This will run an experiment with all the default configurations on a tiny bigearthnet subset.
It will automatically download a small dataset, will train a shallow baseline model end-to-end for 3 epochs.
This should run just fine on a CPU in a few minutes and will ensure everything is properly installed.

# Dataset
This project uses BigEarthNet Sentinel-2 Image Patches from [bigearthnet](https://bigearth.net/). For more in-depth information about the original dataset, you can read the [release paper](https://bigearth.net/static/documents/BigEarthNet_IGARSS_2019.pdf).


For convenience, the raw data has already been converted into [Hub datasets](https://docs.activeloop.ai/datasets) which are provided with this repo.  In these hub datasets, we are only considering bands 2,3,4 of the original spectral data, which (roughly) correspond to the B,G and R channels.

To view how the data was prepared, head to [/bigearthnet/data/scripts/](/bigearthnet/data/scripts/).

3 versions of the dataset have been constructed using the BGR bands:


| Name               | Size   |
| ---                | ---    |
| Bigearthnet-mini   | 9.3 MB |
| Bigeathnet-medium  | 2.5 GB |
| Bigeartnet-full    | 30 GB  |

* **bigearthnet-mini**: A tiny subset of the original data meant for debugging code and functionality. Used for running end-to-end tests during github actions. It is composed of 90 train samples, 30 validation samples and 30 test samples.

* **bigearthnet-medium**: Composed of ~10% of the bigearthnert data. It is meant to be used to train models on a smaller scale and run vast hyper-parameter searches in reasonable time/compute. It is composed of 25 000 train samples, 5000 validation samples and 5000 test samples.

* **bigearthnet-full**: The full dataset. It is composed of 269 695 train samples, 123 723 validation samples and 125 866 test samples. Splits were obtained from [here](https://git.tu-berlin.de/rsim/BigEarthNet-S2_43-classes_models/-/tree/master/splits) 

## Getting the data

All data is hosted on a shared [google drive](https://drive.google.com/drive/folders/1lXv1LB1lfdpHHVSx4Mwj_cZDYmWYGRSa). 

You do not need to download the files manually.
By default, the `bigearthnet-mini` dataset will automatically be downloaded and extracted to the `datasets/` folder when first running an experiment. 

To use another dataset version, simply overwrite the `datamodule.dataset_name` parameter. For example, to run an experiment using the `bigearthnet-medium` dataset, simply run:

    python train.py ++datamodule.dataset_name=bigearthnet-medium

You can specify a different download folder by overriding the `datamodule.dataset_dir` directory. 

# Training Models

To train a model, simply run:

    python train.py 

This will launch an experiment with all of the default parameters.

## Overriding Parameters

This project uses [hydra](hydra.cc) to manage configuration files.

A default configuration can be found under the [configs](bigearthnet/configs/) directory.

To specify different parameters, you can simply override the appropriate parameter from the command line.
For example, to train with a [TIMM](https://github.com/rwightman/pytorch-image-models) resnet34 pretrained model, a learning rate of 0.001 and adam optimizer, run the following:

    python train.py model=timm ++model.model_name=resnet34 ++model.pretrained=true ++config.optimizer.name='adam' ++config.optimizer.lr=0.001 ++max_epochs=100

## Hyper-parameter search

To perform hyper-parameter search, we can run a grid-search over common parameters using the `--multirun` hydra flag. For example, we will sweep a bunch of different learning rates:

    python train.py --multirun ++config.optimizer.name='adam','sgd' ++config.optimizer.lr=0.1,0.01,0.001,0.0001

There exist many great tools for doing more advanced hyper-parameter searching, and hydra plugins easy ways to extend this support.

# Tensorboard

After training models, you can view logged stats in tensorboard. By default, all experiments get saved under the `outputs/` folder.
Simply run

    tensorboard --logdir outputs/

Then, open a browser and head to [http://localhost:6006/](http://localhost:6006/) to view experiments.

This repo supports hyper parameter view under `hparams` tab, evolution of confusion matrices under `images` tab and profiling of the code under `pytorch profiler` tab.


# Model Evaluation

To evaluate a model on the test set, specify the output directory of the trained model, e.g.:

    python eval.py --config-path outputs/bigearthnet-mini/default_group/2022-09-23T16:30:22/baseline_lr_0.0001_adam/lightning_logs/version_0/

This will automatically load the best model checkpoint and produce a summary of all the metrics on that dataset's test set (e.g. if it was trained on `bigearthnet-mini`, the evaluation will be on the `bigearthnet-mini` test set). Results of the evaluation will be in the same folder.

You can also evaluate on other test datasets than the one that was trained on, e.g. evaluate on the medium test set even though we trained on mini:

    python eval.py --config-path outputs/bigearthnet-mini/default_group/2022-09-23T16:30:22/baseline_lr_0.0001_adam/lightning_logs/version_0 ++datamodule.dataset_name="bigearthnet-medium"

