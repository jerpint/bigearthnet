import pandas as pd
import os
import subprocess

BASE_URL = "https://git.tu-berlin.de/rsim/BigEarthNet-S2_43-classes_models/-/raw/master/splits/"
SEED = 42
DEBUG_SAMPLES = 100
split_files = ["train.csv", "val.csv", "test.csv"]


def download_from_url(url: str, dst: str):
    """Download a file from url to dst."""
    subprocess.run(["curl", url, "-o", dst], check=True)


def generate_debug_split(split_fname, debug_fname, debug_samples, seed=None):
    """Returns n_samples from a split."""
    debug_df = pd.read_csv(split_fname).sample(debug_samples, random_state=seed)
    debug_df.to_csv(debug_fname, index=False)


def download_original_splits(splits_dir: str):
    """Download the splits from the original repo."""
    if not os.path.isdir(splits_dir):
        os.mkdir(splits_dir)

    for fname in split_files:
        url = BASE_URL + fname
        dst = str(os.path.join(splits_dir, fname))
        if not os.path.exists(dst):
            download_from_url(url, dst)


def generate_debug_splits(splits_dir):
    for fname in split_files:
        split_path = os.path.join(splits_dir, fname)
        debug_path = os.path.join(splits_dir, "debug_" + fname)

        if not os.path.exists(debug_path):
            generate_debug_split(
                split_path, debug_path, debug_samples=DEBUG_SAMPLES, seed=SEED
            )


if __name__ == "__main__":
    download_original_splits(splits_dir="splits")
    generate_debug_splits(splits_dir="splits")
