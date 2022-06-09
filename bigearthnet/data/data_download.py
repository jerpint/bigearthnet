"""Generate a small debug dataset from the original BigEarth data.

This script assumes the original data was already downloaded and extracted.
It will then fetch the train, val and test splits from an external source, and
sample at random from that data to create a small debugging dataset. The debug
dataset can be used to work and test everything locally.

Example usage:

        $ python data_download.py --splits-dir "splits/" --output-dir "debug_dataset/" --dataset-root-dir $SCRATCH/bigearth/BigEarthNet-v1.0 --tar

"""

import argparse
import os
import subprocess
from shutil import copytree

import pandas as pd

SEED = 42
BASE_URL = "https://git.tu-berlin.de/rsim/BigEarthNet-S2_43-classes_models/-/raw/master/splits/"
num_debug_samples = [100, 10, 10]
splits = ["train.csv", "val.csv", "test.csv"]


def download_from_url(url: str, dst: str):
    """Download a file from url to dst."""
    subprocess.run(["curl", url, "-o", dst], check=True)


def download_splits(splits_dir: str, splits):
    """Download the splits from the original repo."""
    if not os.path.isdir(splits_dir):
        os.mkdir(splits_dir)

    for split in splits:
        url = BASE_URL + split
        dst = str(os.path.join(splits_dir, split))
        if not os.path.exists(dst):
            download_from_url(url, dst)


def generate_debug_split(split_fname, debug_fname, num_samples, seed=None):
    """Returns n_samples from a split."""
    debug_df = pd.read_csv(split_fname, header=None).sample(num_samples, random_state=seed)
    debug_df.to_csv(debug_fname, index=False, header=False)


def generate_debug_splits(splits_dir, debug_splits_dir, splits, num_debug_samples, seed):
    if not os.path.isdir(debug_splits_dir):
        os.mkdir(debug_splits_dir)

    for split, num_samples in zip(splits, num_debug_samples):
        split_fname = os.path.join(splits_dir, split)
        debug_fname = os.path.join(debug_splits_dir, split)

        generate_debug_split(
            split_fname, debug_fname, num_samples=num_samples, seed=seed
        )


def generate_debug_dataset(debug_splits_dir, splits, dataset_root_dir, output_dir, tar):
    print("Generating debug dataset...")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    for split in splits:
        split_fname = os.path.join(debug_splits_dir, split)
        assert os.path.exists(split_fname), f"{split_fname} not found."

        folders = pd.read_csv(split_fname, header=None)[0].to_list()  # Each row in the csv is a folder name
        for folder in folders:
            src = os.path.join(dataset_root_dir, folder)
            dst = os.path.join(output_dir, folder)
            copytree(src, dst, dirs_exist_ok=True)

    if tar:
        subprocess.run(["tar", "zcvf", "BigEarthNet-v1.0-DEBUG.tar", str(output_dir)], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", help="Directory to download origional splits to.")
    parser.add_argument("--output-dir", help="Directory to save the debug dataset to.")
    parser.add_argument("--dataset-root-dir", help="Root directory of original extracted dataset.")
    parser.add_argument("--tar", help="Save the debug dataset to an archive", action="store_true")
    args = parser.parse_args()

    splits_dir = args.splits_dir
    debug_splits_dir = os.path.join(splits_dir, "debug/")
    output_dir = args.output_dir
    dataset_root_dir = args.dataset_root_dir

    download_splits(
        splits_dir=splits_dir,
        splits=splits
    )

    generate_debug_splits(
        splits_dir=splits_dir,
        debug_splits_dir=debug_splits_dir,
        splits=splits,
        num_debug_samples=num_debug_samples,
        seed=SEED,
    )

    generate_debug_dataset(
        debug_splits_dir=debug_splits_dir,
        splits=splits,
        dataset_root_dir=dataset_root_dir,
        output_dir=output_dir,
        tar=True
    )
