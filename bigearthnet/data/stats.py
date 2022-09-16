import json
import pprint

import numpy as np
import torch
from tqdm import tqdm


def compute_class_counts(onehot_labels: np.ndarray) -> np.ndarray:
    """Given a collection of onehot labels, compute the number of positives for each class."""
    num_classes = len(
        onehot_labels[0]
    )  # use first instance to determine number of classes
    class_positives = np.zeros((num_classes,))
    for onehot in onehot_labels:
        assert len(onehot) == num_classes
        class_positives += onehot
    return class_positives


def compute_class_weights(
    positive_counts: np.ndarray,
    num_samples: int,
    a_min=None,
    a_max=None,
):
    """Compute class weights based on their frequency in the dataset, useful for re-weighing the loss function.

    For example, if a dataset contains 100 positive and 300 negative examples of a single class, then class_weight
    for the class should be equal to 300/100 = 3.
    """

    assert num_samples >= max(
        positive_counts
    ), "can't have less samples than the top count."
    negative_counts = num_samples - positive_counts
    class_weights = negative_counts / positive_counts
    if any([a is not None for a in [a_min, a_max]]):
        # only clip if either a_min or a_max are specified
        class_weights = np.clip(class_weights, a_min=a_min, a_max=a_max)
    return class_weights


def save_json_dict(json_dict, output_fname):
    """Save json_dict to output_fname."""
    pprint.pprint(json_dict)
    with open(output_fname, "w") as f:
        json.dump(json_dict, f, indent=4)


def compute_dataloader_mean_std(pt_dataloader, num_channels=3):
    """
    Compute the mean and std along image channels.

    Takes in a pytorch dataloader, and loads the data batch by batch.

    See this post for more details:
    https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html
    """
    px_sum = torch.zeros(num_channels, dtype=torch.float)
    px_sum_sq = torch.zeros(num_channels, dtype=torch.float)

    for batch in tqdm(pt_dataloader):
        if isinstance(batch, dict):
            # hub dataset format
            imgs = batch["data"]
        elif isinstance(batch, list):
            # torchvision dataset format
            imgs = batch[0]
        else:
            raise NotImplementedError()

        px_sum += torch.sum(imgs, axis=(0, 2, 3))
        px_sum_sq += torch.sum(imgs**2, axis=(0, 2, 3))

    # compute total pixel count
    img_count = len(pt_dataloader.dataset)
    px_count = img_count * imgs.shape[2] * imgs.shape[3]  # num_images * H * W

    # mean and std
    channel_mean = px_sum / px_count
    channel_var = (px_sum_sq / px_count) - (channel_mean**2)
    channel_std = np.sqrt(channel_var)

    return channel_mean, channel_std


if __name__ == "__main__":
    from bigearthnet.datamodules.bigearthnet_datamodule import (
        BigEarthNetDataModule,
        hub_labels_to_onehot,
    )

    dataset_dir = "../data/"  # root directory where to download the datasets
    dataset_name = "bigearthnet-medium"  # One of bigearthnet-mini, bigearthnet-medium, bigearthnet-full
    batch_size = 16

    dm = BigEarthNetDataModule(dataset_dir, dataset_name, batch_size=batch_size)
    dm.setup()
    ds = dm.train_dataloader().dataset
    n_classes = len(ds.class_names)
    num_samples = len(ds)
    class_names = ds.class_names
    hub_labels_list = ds.dataset.labels[:].numpy(aslist=True)
    onehot_labels = [
        hub_labels_to_onehot(hub_label, n_classes) for hub_label in hub_labels_list
    ]

    # compute class weights
    positive_counts = compute_class_counts(onehot_labels)
    class_weights = compute_class_weights(
        positive_counts, num_samples, a_min=1, a_max=100
    )
    json_dict = {name: weight for name, weight in zip(ds.class_names, class_weights)}
    save_json_dict(json_dict, output_fname="class_weights.json")

    # Compute the channel mean and std of the dataset
    pt_dataloader = dm.train_dataloader()
    mean, stddev = compute_dataloader_mean_std(pt_dataloader)

    json_dict = {
        "mean": mean.tolist(),
        "std": stddev.tolist(),
    }
    save_json_dict(json_dict, output_fname="mean_stddev.json")
