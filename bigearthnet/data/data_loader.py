import logging
import pathlib
import typing

import hub
import numpy as np
import pytorch_lightning as pl
import torch.utils.data.dataset
import torch.utils.data.dataloader

logger = logging.getLogger(__name__)


class HubParser(torch.utils.data.dataset.Dataset):
    """Dataset parser class used to iterate over the BigEarthNet-S2 data."""

    def __init__(
        self,
        dataset_path_or_object: typing.Union[typing.Union[typing.AnyStr, pathlib.Path], hub.Dataset],
        **extra_hub_kwargs,
    ):
        """Initialize the BigEarthNet-S2 hub dataset parser (in read-only mode)."""
        if isinstance(dataset_path_or_object, hub.Dataset):
            assert not extra_hub_kwargs, "dataset is already opened, can't use kwargs"
            self.dataset = dataset_path_or_object
        else:
            self.dataset = hub.load(str(dataset_path_or_object), read_only=True, **extra_hub_kwargs)

    def __len__(self) -> int:
        """Returns the total size (patch count) of the dataset."""
        return len(self.dataset)

    def __getitem__(self, item: int) -> typing.Dict[str, typing.Any]:
        """Returns a single data sample loaded from the dataset.

        For BigEarthNet, the data sample simply consists of the patch data and its labels. The
        patch data and labels will be converted from their original types to float32 and int16,
        respectively, in order to make sure that PyTorch will be able to batch them.
        """
        data = self.dataset[int(item)]  # cast in case we're using numpy ints or something similar
        assert tuple(self.tensor_names) == ("patches/data", "patches/labels")
        class_labels = np.zeros((len(self.class_names), ), dtype=np.int16)
        class_labels[data["patches/labels"].numpy()] = 1
        return {
            "patches/data": data["patches/data"].numpy().astype(np.float32),
            "patches/labels": class_labels,
        }

    def summary(self) -> None:
        """Forwards the call to print a summary of the dataset."""
        return self.dataset.summary()

    def visualize(self, *args, **kwargs):
        """Forwards the call to show the dataset content (notebook-only)"""
        return self.dataset.visualize(*args, **kwargs)

    @property
    def dataset_info(self) -> typing.Dict[str, typing.Any]:
        """Returns metadata information parsed from the hub dataset object."""
        return dict(self.dataset.info)

    @property
    def dataset_name(self) -> typing.AnyStr:
        """Returns the dataset name used to identify this particular dataset."""
        return self.dataset_info["name"]

    @property
    def class_names(self) -> typing.List[str]:
        """Returns the list of class names that correspond to the label indices in the dataset."""
        return list(self.dataset.info.class_names)

    @property
    def tensor_info(self) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
        """Returns the dictionary of tensor info objects (hub-defined) parsed from the dataset.

        The returned objects can help downstream processing stages figure out what kind of data
        they will be receiving from this parser.
        """
        return {k: v.info for k, v in self.dataset.tensors.items()}

    @property
    def tensor_names(self) -> typing.List[str]:
        """Names of the tensors that will be provided in the loaded data samples.

        Note that additional tensors and other attributes may be loaded as well, but these are the
        'primary' fields that should be expected by downstream processing stages.
        """
        return list(self.tensor_info.keys())


class DataModule(pl.LightningDataModule):
    """Data module class that prepares BigEarthNet-S2 dataset parsers and instantiates data loaders."""

    def __init__(
        self,
        dataset_path: typing.Union[typing.AnyStr, pathlib.Path],
        batch_size: int,
        num_workers: int = 0,
        split_seed: int = 0,
        **extra_hub_kwargs,
    ):
        """Validates the hyperparameter config dictionary and sets up internal attributes."""
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.split_seed = split_seed
        self._data_parser = None
        self._extra_hub_kwargs = extra_hub_kwargs
        self.train_parser, self.valid_parser, self.test_parser = None, None, None

    def prepare_data(self):
        """Downloads/extracts/unpacks the data if needed (we don't)."""
        pass

    def _get_80_10_10_split_indices(self, dataset_size: int) -> typing.Tuple:
        """Returns the sample indices for a 80-10-10 (train/valid/test) split."""
        rng = np.random.default_rng(self.split_seed)
        shuffled_idxs = rng.permutation(dataset_size)
        eval_set_size = int(0.2 * dataset_size)
        valid_set_size = eval_set_size // 2
        train_set_size = dataset_size - eval_set_size
        return (
            shuffled_idxs[0:train_set_size],
            shuffled_idxs[train_set_size:(train_set_size + valid_set_size)],
            shuffled_idxs[(train_set_size + valid_set_size):],
        )

    def setup(self, stage=None) -> None:
        """Parses and splits all samples across the train/valid/test parsers."""
        if self._data_parser is None:
            self._data_parser = HubParser(self.dataset_path, **self._extra_hub_kwargs)
        train_idxs, valid_idxs, test_idxs = self._get_80_10_10_split_indices(len(self._data_parser))
        assert len(np.intersect1d(train_idxs, valid_idxs)) == 0
        assert len(np.intersect1d(train_idxs, test_idxs)) == 0
        assert len(np.intersect1d(valid_idxs, test_idxs)) == 0
        if stage == "fit" or stage is None:
            if self.train_parser is None:
                self.train_parser = torch.utils.data.dataset.Subset(self._data_parser, train_idxs)
            if self.valid_parser is None:
                self.valid_parser = torch.utils.data.dataset.Subset(self._data_parser, valid_idxs)
        if stage == "test" or stage is None:
            if self.test_parser is None:
                self.test_parser = torch.utils.data.dataset.Subset(self._data_parser, test_idxs)

    def train_dataloader(self) -> torch.utils.data.dataloader.DataLoader:
        """Creates the training dataloader using the training data parser."""
        assert self.train_parser is not None, "must call 'setup' first!"
        return torch.utils.data.dataloader.DataLoader(
            dataset=self.train_parser,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> torch.utils.data.dataloader.DataLoader:
        """Creates the validation dataloader using the validation data parser."""
        assert self.valid_parser is not None, "must call 'setup' first!"
        return torch.utils.data.dataloader.DataLoader(
            dataset=self.valid_parser,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> torch.utils.data.dataloader.DataLoader:
        """Creates the testing dataloader using the testing data parser."""
        assert self.test_parser is not None, "must call 'setup' first!"
        return torch.utils.data.dataloader.DataLoader(
            dataset=self.test_parser,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.NOTSET)
    hub_dataset_path = pathlib.Path("/wdata/datasets/bigearthnet-mini/.hub_dataset/")
    data_module = DataModule(dataset_path=hub_dataset_path, batch_size=16)
    data_module.setup()
    train_data_loader = data_module.train_dataloader()
    assert len(train_data_loader) > 0
    minibatch = next(iter(train_data_loader))
    assert "patches/data" in minibatch and len(minibatch["patches/data"]) <= 16
    assert "patches/labels" in minibatch and len(minibatch["patches/labels"]) <= 16
    print("all done")
