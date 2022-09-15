import numpy as np
from bigearthnet.utils.stats import compute_class_counts, compute_class_weights


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
