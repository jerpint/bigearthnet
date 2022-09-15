import json
import pprint
import numpy as np


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
        # only clip if a_min and/or a_max are specified
        class_weights = np.clip(class_weights, a_min=a_min, a_max=a_max)  # clamp values
    return class_weights


def save_class_weights(class_weights, class_names, output_fname):
    """Save computed class weights to a json dict."""
    json_output = {name: weight for name, weight in zip(class_names, class_weights)}
    pprint.pprint(json_output)
    with open(output_fname, "w") as f:
        json.dump(json_output, f, indent=4)


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

    # compute counts
    positive_counts = compute_class_counts(onehot_labels)
    class_weights = compute_class_weights(
        positive_counts, num_samples, a_min=1, a_max=100
    )
    save_class_weights(class_weights, class_names, output_fname="class_weights.json")
