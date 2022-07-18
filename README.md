# BigEarthNet


BigEarthNet classification


# Local Setup

## Install the dependencies:
It is recommended to work in a virtual environment. 
Once activated, run:

    pip install -e .


## Test your installation
To test your install, cd into bigearthnet/ and run:

    cd bigearthnet/
    python train.py

This will run an experiment with all the default configurations. 
It will automatically download a small debugging dataset, and will run an experiment end to end for 5 epochs.
This step shouldn't take longer than a few minutes and will ensure everything is properly installed.

## Specifying experiment parameters

This project uses [hydra](hydra.cc) to manage configurtion files.
To specify additonal configurations, do so simply from the command line.
For example, to train with a [timm](https://github.com/rwightman/pytorch-image-models) resnet34 pretrained model, a learning rate of 0.001 and adam optimizer, run the following:

    python train.py model=timm ++model.model_name=resnet34 ++model.pretrained=true ++config.optimizer='adam' ++config.optimizer=0.001 ++max_epochs=10


# Data

This project uses BigEarthNet with Sentinel-2 Image Patches from [bigearthnet](https://bigearth.net/). For more in-depth information about the data itself, you can read the [release paper](https://bigearth.net/static/documents/BigEarthNet_IGARSS_2019.pdf).
For convenience, the raw data has already been converted into [Hub datasets](https://docs.activeloop.ai/datasets).

There are currently 3 versions of the dataset composed of the BGR bands:
* bigearthnet-mini: A small dataset used for working locally and quickly iteration. 
It is composed of 80 train samples, 10 validation samples and 10 test samples.
* bigearthnet-medium: A medium sized dataset which can be used to train models and observe behaviour when some signal passes through.
It is composed of 4000 train samples, 500 validation samples and 500 test samples.
* bigearthnet-full [COMING SOON]

Each version can be specified using the configuration files. For example, to run an experiment using the bigearthnet-medium dataset, simply run:

    python train.py ++datamodule.dataset_name=bigearthnet-medium

The dataset will automatically be downloaded and extracted to `datasets/` and the script will find and use the proper data.


