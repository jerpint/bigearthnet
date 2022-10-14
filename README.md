# BigEarthNet 🌎

Welcome to the [bigearthnet](https://bigearth.net/) classification repo!  
This project was built in the context of an [applied deep learning workshop in computer vision.](https://catalogue.ivado.umontreal.ca/Web/MyCatalog/ViewP?id=bfBsk1MpdKCuFXDwCrvZMw%3d%3d&pid=DwpGfXsYFQ5dNLAWEt9mWQ%3d%3d&mc_cid=c7c86767f2&mc_eid=3258909e75)
This repo showcases modern tools and libraries for applied deep learning. 
Accompanying slides and explanations can be found [here](https://docs.google.com/presentation/d/1uIAV55ZLbQafmiDmHCMeWZPEdmzWzWIm8pTKE5Z2YUw/edit?usp=sharing).

Here are a few of the features baked-in to this repo:

* [Pytorch-Lightning](https://www.pytorchlightning.ai/): Implements all the training loops and boilerplate code
* [Hydra](https://hydra.cc/): Easily manage and configure experiment parameters
* [TIMM](https://github.com/rwightman/pytorch-image-models): A model zoo of SOTA pre-trained classification models
* [Tensorboard](https://www.tensorflow.org/tensorboard): Logger used to track experiment progress
* [Deep Lake / Activeloop Hub](https://github.com/activeloopai/deeplake): An efficient dataset/dataloader manager (think HDF5 but deep-learning centric).

The focus of this repository is centered around model training and evaluation. Deployment is not considered in this project.

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
It will automatically download a small dataset and train a shallow baseline model end-to-end for 3 epochs.
This should run on a CPU (<1 minute) and will ensure everything is properly installed.

# Dataset

This project uses BigEarthNet Sentinel-2 Image Patches from [bigearthnet](https://bigearth.net/). For more in-depth information about the original dataset, you can read the [release paper](https://bigearth.net/static/documents/BigEarthNet_IGARSS_2019.pdf).
For convenience, the raw data has already been converted into [Hub datasets](https://docs.activeloop.ai/datasets) which are provided with this repo.  In these hub datasets, we are only considering bands 2,3,4 of the original spectral data, which (roughly) correspond to the B,G and R channels.
To view how the data was prepared, head to [/bigearthnet/data/scripts/](/bigearthnet/data/scripts/).

3 versions of the dataset have been constructed using the BGR bands of the original dataset:

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

A tensorboard summary of the best models can be found [here](https://tensorboard.dev/experiment/2M0cvnH5S4uaikOJKYkaBw/#scalars)

# Model Evaluation

By default, all outputs can be found under the `outputs/` folder. 
Each run will be timestamped, and contain a `checkpoints` directory with a `last-model.ckpt` and `best-model.ckpt`.

To evaluate a model on the test set, run the `eval.py` script while specifying the checkpoint of the best trained model, e.g.:

    python eval.py --ckpt-path /path/to/best-model.ckpt

This will automatically load the model checkpoint and produce a summary of all the metrics on a given test set. 
Results of the evaluation will be saved where the script was run.
You can specify which test set to evaluate on with the `--dataset-name` flag (by default it evaluates on `bigearthnet-mini`).
This is useful for e.g. training on `bigearthnet-medium` and evaluating on `bigearthnet-full` test set. 

    python eval.py --ckpt-path /path/to/best-model.ckpt --dataset-name bigearthnet-full

For additional parameters such as speciying to do the evaluation on a gpu, run:

    python eval.py --help for additional parameters.

# Sample Notebook

You can view a sample notebook [here](https://colab.research.google.com/drive/1ijpM9RmvfUaBkfHsdgphmkdl9EY8BLSp#scrollTo=IlUOy0wEljwz).
You can also follow the setup in the notebook to run models from within colab. 

Note that viewing results within tensorboard won't be possible from colab, but you can download the `outputs/`  folder locally to view them.
You can also use pre-trained models from within colab.

# Pretrained models

A tensorboard summary of the best models can be found [here](https://tensorboard.dev/experiment/2M0cvnH5S4uaikOJKYkaBw/#scalars)

You can download the pre-trained models here:

Pre-trained [ConvNext](https://drive.google.com/file/d/1EyLDVoZKK-GZNnr_VEHNoH9eJoVbi2Nx/view?usp=sharing)
Pre-trained [ViT](https://drive.google.com/file/d/1uVPSDAaDnEUDoa4fCshHXgwYORaeSLt6/view?usp=sharing)
