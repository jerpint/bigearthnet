import logging
import pathlib
import typing

import hub
import numpy as np
import pytorch_lightning as pl
import torch.utils.data.dataloader
import torch.utils.data.dataset
from hydra.utils import instantiate
from torchvision import transforms

logger = logging.getLogger(__name__)


class HubDataset(torch.utils.data.dataset.Dataset):
    """Dataset class used to iterate over the BigEarthNet-S2 data."""

    def __init__(
        self,
        dataset_path: pathlib.Path,
        transforms=None,
        **extra_hub_kwargs,
    ):
        """Initialize the BigEarthNet-S2 hub dataset (in read-only mode)."""
        self.dataset_path = dataset_path
        self.dataset = hub.load(self.dataset_path, read_only=True, **extra_hub_kwargs)
        self.transforms = transforms

    def __len__(self) -> int:
        """Returns the total size (patch count) of the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> typing.Dict[str, typing.Any]:
        """Returns a single data sample loaded from the dataset.

        For BigEarthNet, the data sample simply consists of the patch data and its labels. The
        patch data and labels will be converted from their original types to float32 and int16,
        respectively, in order to make sure that PyTorch will be able to batch them.
        Labels are converted to one-hot representation.
        """
        item = self.dataset[
            int(idx)
        ]  # cast in case we're using numpy ints or something similar
        assert tuple(self.tensor_names) == ("data", "labels")

        labels_idx = item["labels"].numpy()
        onehot_labels = np.zeros((len(self.class_names),), dtype=np.int16)
        onehot_labels[labels_idx] = 1
        labels = torch.tensor(onehot_labels)

        img_data = item["data"].numpy().astype(np.float32)
        img_data = torch.tensor(img_data)

        if self.transforms:
            img_data = self.transforms(img_data)

        return {
            "data": img_data,
            "labels": labels,
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
        dataset_path: str,
        batch_size: int,
        num_workers: int = 0,
        transforms=None,
        **extra_hub_kwargs,
    ):
        """Validates the hyperparameter config dictionary and sets up internal attributes."""
        super().__init__()
        self.dataset_path = pathlib.Path(dataset_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._extra_hub_kwargs = extra_hub_kwargs
        self.train_dataset, self.valid_dataset, self.test_dataset = None, None, None
        self.transforms = transforms

    def prepare_data(self):
        """Downloads/extracts/unpacks the data if needed."""
        pass

    def setup(self, stage=None) -> None:
        """Parses and splits all samples across the train/valid/test datasets."""
        if stage == "fit" or stage is None:
            if self.train_dataset is None:
                self.train_dataset = HubDataset(
                    self.dataset_path / "train",
                    transforms=self.transforms,
                    **self._extra_hub_kwargs,
                )
            if self.valid_dataset is None:
                self.valid_dataset = HubDataset(
                    self.dataset_path / "val",
                    transforms=self.transforms,
                    **self._extra_hub_kwargs,
                )
        if stage == "test" or stage is None:
            if self.test_dataset is None:
                self.test_dataset = HubDataset(
                    self.dataset_path / "test",
                    transforms=self.transforms,
                    **self._extra_hub_kwargs,
                )

    def train_dataloader(self) -> torch.utils.data.dataloader.DataLoader:
        """Creates the training dataloader using the training dataset."""
        assert self.train_dataset is not None, "must call 'setup' first!"
        return torch.utils.data.dataloader.DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> torch.utils.data.dataloader.DataLoader:
        """Creates the validation dataloader using the validation data parser."""
        assert self.valid_dataset is not None, "must call 'setup' first!"
        return torch.utils.data.dataloader.DataLoader(
            dataset=self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> torch.utils.data.dataloader.DataLoader:
        """Creates the testing dataloader using the testing data dataset."""
        assert self.test_dataset is not None, "must call 'setup' first!"
        return torch.utils.data.dataloader.DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


def load_transforms(cfg):
    if cfg.get("transforms"):
        return transforms.Compose([instantiate(T) for T in cfg.transforms])
    return None


def load_datamodule(cfg):
    transforms = load_transforms(cfg)
    data_module = DataModule(transforms=transforms, **cfg.datamodule)
    data_module.setup()
    return data_module


if __name__ == "__main__":
    # TODO: Put this as a test instead
    logging.basicConfig()
    logging.getLogger().setLevel(logging.NOTSET)

    hub_dataset_path = pathlib.Path("./debug_dataset/hub_dataset/")
    dm = load_datamodule(hub_dataset_path, batch_size=16)

    train_data_loader = dm.train_dataloader()
    assert len(train_data_loader) > 0
    minibatch = next(iter(train_data_loader))
    assert "data" in minibatch and len(minibatch["data"]) <= 16
    assert "labels" in minibatch and len(minibatch["labels"]) <= 16
    assert minibatch["labels"].shape[1] == 43
    print("all done")
