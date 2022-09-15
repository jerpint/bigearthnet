import numpy as np
from bigearthnet.utils.stats import (
    compute_class_counts,
    compute_class_weights,
    compute_dataloader_mean_std,
)

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset


def test_compute_class_counts():

    # Test
    onehot_labels = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 2],
        ]
    )

    positive_counts = compute_class_counts(onehot_labels)
    expected_positive_counts = np.array([1, 1, 1, 3])

    assert np.equal(positive_counts, expected_positive_counts).all()


def test_compute_class_weights():

    # Test
    positive_counts = np.array([100, 400, 10, 50])
    num_samples = 400

    class_weights = compute_class_weights(
        positive_counts, num_samples=num_samples, a_min=None, a_max=None
    )
    expected_class_weights = np.array([3, 0, (400 - 10) / 10, (400 - 50) / 50])

    assert np.equal(class_weights, expected_class_weights).all()


def test_compute_dataloader_mean_std():
    def transform(x):
        return np.expand_dims(x, axis=0).copy().astype("float")

    mnist_dataset = torchvision.datasets.MNIST(
        root=".", train=False, download=True, transform=transform
    )
    mnist_dataloader = DataLoader(mnist_dataset, batch_size=32)

    # computed with batch stats
    computed_mean, computed_std = compute_dataloader_mean_std(
        mnist_dataloader, num_channels=1
    )

    # computed without batch stats
    mnist_mean = torch.mean(mnist_dataset.data.view(-1).type(torch.float))
    mnist_std = torch.std(mnist_dataset.data.view(-1).type(torch.float))

    assert torch.isclose(computed_mean, mnist_mean)
    assert torch.isclose(computed_std, mnist_std)
