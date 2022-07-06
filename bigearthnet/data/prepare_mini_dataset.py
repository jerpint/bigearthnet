"""Generate a mini dataset from the original BigEarth data.

This script assumes the original data was already downloaded and extracted.
It will then fetch the train, val and test splits from an external source, and
sample at random from that data to create a small debugging dataset. The debug
dataset can be used to work and test everything locally.

Example usage:

        $ python prepare_mini_dataset.py \
        --splits-dir "$SCRATCH/bigearth/splits/" \
        --output-dir "$SCRATCH/bigearth/bigearth-mini/" \
        --dataset-root-dir $SCRATCH/bigearth/BigEarthNet-v1.0 \
        --split-samples 160 20 20

"""

import argparse
import os
import pathlib
import subprocess
from shutil import copytree

import pandas as pd
from tqdm import tqdm

BASE_URL = "https://git.tu-berlin.de/rsim/BigEarthNet-S2_43-classes_models/-/raw/master/splits/"
SPLITS = ["train", "val", "test"]


def download_from_url(url: str, dst: str):
    """Download a file from url to dst."""
    subprocess.run(["curl", url, "-o", dst], check=True)


def sample_from_csv(csv, num_samples, seed=None):
    """Randomly sample num_samples rows from a csv."""
    return pd.read_csv(csv, header=None).sample(num_samples, random_state=seed)


def download_splits(splits_dir: str):
    """Download the splits from $BASE_URL to splits_dir."""
    if not os.path.isdir(splits_dir):
        os.mkdir(splits_dir)

    for split in SPLITS:
        url = BASE_URL + split + ".csv"
        dst = str(os.path.join(splits_dir, split + ".csv"))
        if not os.path.exists(dst):
            print(f"Downloading {split}.csv to {os.path.abspath(splits_dir)}\n")
            download_from_url(url, dst)


def generate_mini_splits(splits_dir, output_dir, split_samples, seed):
    """Generates splits (train.csv, val.csv, test.csv) by random sampling the original split files.

    Files are saved in $output_dir/splits/ as csv files.
    """
    mini_splits_dir = os.path.join(output_dir, "splits/")
    if not os.path.isdir(mini_splits_dir):
        os.makedirs(mini_splits_dir)

    for split, num_samples in zip(SPLITS, split_samples):
        print(f"Sampling {num_samples} samples for {split}")
        split_fname = os.path.join(splits_dir, split + ".csv")
        mini_split_fname = os.path.join(mini_splits_dir, split + ".csv")

        sampled_split = sample_from_csv(split_fname, num_samples=num_samples, seed=seed)
        sampled_split.to_csv(mini_split_fname, index=False, header=False)


def generate_mini_dataset(dataset_root_dir, output_dir):
    """Creates a new dataset comprised only of the samples referenced in the debug splits."""
    print(f"Generating mini dataset in {os.path.abspath(output_dir)}...")
    output_dir = pathlib.Path(output_dir)
    mini_splits_dir = os.path.join(output_dir, "splits/")
    mini_data_dir = os.path.join(output_dir, "data/")
    if not os.path.isdir(mini_data_dir):
        os.makedirs(mini_data_dir)

    for split in SPLITS:
        print(f"copying sample {split} folders...")
        split_fname = os.path.join(mini_splits_dir, split + ".csv")
        split_df = pd.read_csv(split_fname, header=None)
        folders = split_df[0].to_list()  # Each row in the csv is a folder name
        for folder in tqdm(folders):
            # Copy the folder from the original dataset to the debug dataset
            src = os.path.join(dataset_root_dir, folder)
            dst = os.path.join(mini_data_dir, folder)
            copytree(src, dst, dirs_exist_ok=True)

    # Compress the dataset to a tar archive
    print("Compressing the mini dataset...")
    tar_path = os.path.join(output_dir.parent, "bigearthnet-mini.tar")
    subprocess.run(["tar", "zcf", tar_path, output_dir], check=True)
    print(f"bighearth-mini dataset saved to {os.path.abspath(tar_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits-dir",
        help="Directory to download origional splits to.",
    )
    parser.add_argument("--output-dir", help="Directory to save the debug dataset to.")
    parser.add_argument(
        "--dataset-root-dir",
        help="Root directory of original extracted dataset.",
    )
    parser.add_argument(
        "--split-samples",
        help="Number of samples to use by split [train, valid, test], e.g. 80, 10, 10",
        nargs=3,
        default=[80, 10, 10],
        type=int,
    )
    parser.add_argument(
        "--seed", help="Seed to use for reproducibility", type=int, default=42
    )
    args = parser.parse_args()

    splits_dir = args.splits_dir
    output_dir = args.output_dir
    dataset_root_dir = args.dataset_root_dir
    split_samples = args.split_samples
    seed = args.seed
    print(f"Arguments passed to CLI: {args}\n")

    download_splits(
        splits_dir=splits_dir,
    )

    generate_mini_splits(
        splits_dir=splits_dir,
        output_dir=output_dir,
        split_samples=split_samples,
        seed=seed,
    )

    generate_mini_dataset(
        dataset_root_dir=dataset_root_dir,
        output_dir=output_dir,
    )
