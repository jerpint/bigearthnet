# BigEarthNet

BigEarthNet classification.

This project uses BigEarthNet with Sentinel-2 Image Patches from [bigearthnet](https://bigearth.net/). For more in-depth information about the data itself, you can read the [release paper](https://bigearth.net/static/documents/BigEarthNet_IGARSS_2019.pdf).
For convenience, the raw data has already been converted into [Hub datasets](https://docs.activeloop.ai/datasets).

TODO: Add more context here



# Getting Started

## Clone this project

    cd ~/
    git clone https://github.com/jerpint/bigearthnet

## Install the dependencies:
It is recommended to work in a virtual environment (e.g. conda):

    conda create -n bigearth python=3.8
    conda activate bigeaerth

Once activated, run:

    cd ~/bigearthnet/
    pip install -e .


## Test your installation
To test your install:

    cd ~/bigearthnet/bigearthnet
    python train.py

This will run an experiment with all the default configurations. 
It will automatically download a small debugging dataset, and will run an experiment end to end for 3 epochs.
This step shouldn't take longer than a few minutes and will ensure everything is properly installed.

# Training Models


## Overriding Parameters

This project uses [hydra](hydra.cc) to manage configuration files.
To specify additional configurations, do so simply from the command line.
For example, to train with a [timm](https://github.com/rwightman/pytorch-image-models) resnet34 pretrained model, a learning rate of 0.001 and adam optimizer, run the following:

    python train.py model=timm ++model.model_name=resnet34 ++model.pretrained=true ++config.optimizer.name='adam' ++config.optimizer.lr=0.001 ++max_epochs=10


## Tensorboard

TODO


## Evaluating Models

To evaluate a model on the test set, specify the output directory of the trained model, e.g.:

    python eval.py --config-path outputs/bigearthnet-mini/default_group/2022-09-23T16:30:22/baseline_lr_0.0001_adam/lightning_logs/version_0

Running an evaluation will produce a summary of all the metrics in that same folder on that dataset's test set.
You can also evaluate on other test datasets than the one that was trained on, e.g. on medium even though we trained on small:

    python eval.py --config-path outputs/bigearthnet-mini/default_group/2022-09-23T16:30:22/baseline_lr_0.0001_adam/lightning_logs/version_0 ++datamodule.dataset_name="bigearthnet-medium"


## Datasets
There are currently 3 versions of the dataset composed of the BGR bands:
* bigearthnet-mini: A small dataset used for working locally and quickly iteration. 
It is composed of 80 train samples, 10 validation samples and 10 test samples.
* bigearthnet-medium: A medium sized dataset which can be used to train models and observe behaviour when some signal passes through.
It is composed of 4000 train samples, 500 validation samples and 500 test samples.
* bigearthnet-full: 

Each version can be specified using the configuration files. For example, to run an experiment using the bigearthnet-medium dataset, simply run:

    python train.py ++datamodule.dataset_name=bigearthnet-medium

The dataset will automatically be downloaded and extracted to `bigearthnet/data/` and the script will find and use the proper data.

