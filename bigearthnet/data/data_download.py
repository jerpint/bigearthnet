import pandas as pd
import os
import subprocess

BASE_URL = "https://git.tu-berlin.de/rsim/BigEarthNet-S2_43-classes_models/-/raw/master/splits/"
SEED = 42
num_debug_samples = [500, 50, 50]
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
    debug_df = pd.read_csv(split_fname).sample(num_samples, random_state=seed)
    debug_df.to_csv(debug_fname, index=False)


def generate_debug_splits(splits_dir, splits, num_debug_samples, seed):
    for split, num_samples in zip(splits, num_debug_samples):
        split_fname = os.path.join(splits_dir, split)
        debug_fname = os.path.join(splits_dir, "debug_" + split)

        if not os.path.exists(debug_fname):
            generate_debug_split(
                split_fname, debug_fname, num_samples=num_samples, seed
            )


def generate_debug_dataset(splits_dir, splits, dataset_root_dir, output_dir):
    from shutil import copytree


    for split in splits:
        split_fname = os.path.join(splits_dir, split)
        assert os.path.exists(split_fname), f"{split_fname} not found."

        folders = pd.read_csv(split_fname, header=None)[0].to_list()
        for folder in folders:
            src = os.path.join(dataset_root_dir, folder)
            copytree(src, output_dir, dirs_exist_ok=True)


if __name__ == "__main__":
    splits_dir = "splits"
    outpur_dir = "debug_dataset"
    download_splits(splits_dir=splits_dir, splits=splits)
    generate_debug_splits(
        splits_dir=splits_dir, splits=splits, num_debug_samples=num_debug_samples, seed=SEED
    )
    generate_debug_dataset(splits_dir, splits, dataset_root_dir, output_dir)
