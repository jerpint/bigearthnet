"""Contains the raw dataset parsing and repackaging code for the BigEarthNet(-S2) dataset."""

import argparse
import dataclasses
import datetime
import json
import logging
import pathlib
import pickle
import pprint
import re
import os
import typing

import cv2 as cv
import hub
import pandas as pd
import numpy as np
import tqdm

from bigearthnet.data.prepare_mini_dataset import download_full_splits

logger = logging.getLogger(__name__)

sentinel2_band_names = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"
]
"""Full list of all (12) band names in Sentinel-2 Level 2A tile data.

Note that not all bands will have a 10-meter ground resolution; resizing/resampling might be
required to make a contiguous tensor. See https://gisgeography.com/sentinel-2-bands-combinations
for more information.
"""

sentinel2_band_map = {
    "BGR": ["B02", "B03", "B04"],  # [Blue, Green, Red]
    "BGRNIR": ["B02", "B03", "B04", "B08"],  # [Blue, Green, Red, NIR]
    "ALL": sentinel2_band_names,
}
"""Sentinel-2 Level 2A target band string mappings for ease-of-use while loading the dataset."""

bigearthnet_patch_name_pattern = re.compile(r"^([\w\d]+)_MSIL2A_(\d{8}T(\d+))_(\d+)_(\d+)$")
"""BigEarthNet(-S2) dataset patch folder name pattern.

The first group indicates the Sentinel-2 mission ID that can be either S2A or S2B, the second
group indicates the acquisition datetime string for the Sentinel-2 tile, and the third and fourth
groups indicate the horizontal (column) and vertical (row) indices of the patch in the tile from
which it is extracted, respectively.

See https://bigearth.net/static/documents/Description_BigEarthNet-S2.pdf for more information.
"""

bigearthnet_metadata_field_names = [
    "labels", "coordinates", "projection", "tile_source", "acquisition_date",
]
"""List of metadata field names that we expect to find in the JSON metadata of each patch."""



@dataclasses.dataclass
class BigEarthNetPatch:
    """Defines the (meta)data of a single patch inside the BigEarthNet(-S2) dataset."""

    patch_folder: str
    """Parent folder containing the patch info from BigEarthNet (-S2) dataset."""
    mission_id: str
    """Sentinel-2 mission identifier; should be S2A or S2B."""
    coordinates: typing.Dict[str, float]
    """Dictionary of upper-left and lower-right corner coordinates of the patch."""
    tile_source: str
    """Name of the original unprocessed tile on Copernicus Open Access Hub that contains this patch."""
    tile_row: int
    """Row (vertical) index of the patch inside its source tile."""
    tile_col: int
    """Column (horizontal) index of the patch inside its source tile."""
    acquisition_date: datetime.datetime
    """Acquisition date for the source Sentinel-2 tile."""
    projection: str
    """Projection type for the patch coordinates in Well-Known Text (WKT) format."""
    band_paths_map: typing.Dict[typing.AnyStr, pathlib.Path]
    """Local path to the file containing the data for each of this patch's bands."""
    labels: typing.List[str]
    """List of class label names that are found inside this patch."""

    @property
    def ulx(self) -> float:
        """Returns the X coordinate of the upper-left corner of the patch."""
        return float(self.coordinates["ulx"])

    @ulx.setter
    def ulx(self, x) -> None:
        """Sets the X coordinate of the upper-left corner of the patch."""
        self.coordinates["ulx"] = float(x)

    @property
    def uly(self) -> float:
        """Returns the Y coordinate of the upper-left corner of the patch."""
        return float(self.coordinates["uly"])

    @uly.setter
    def uly(self, y) -> None:
        """Sets the Y coordinate of the upper-left corner of the patch."""
        self.coordinates["uly"] = float(y)

    @property
    def lrx(self) -> float:
        """Returns the X coordinate of the lower-right corner of the patch."""
        return float(self.coordinates["lrx"])

    @lrx.setter
    def lrx(self, x) -> None:
        """Sets the X coordinate of the lower-right corner of the patch."""
        self.coordinates["lrx"] = float(x)

    @property
    def lry(self) -> float:
        """Returns the X coordinate of the lower-right corner of the patch."""
        return float(self.coordinates["lry"])

    @lry.setter
    def lry(self, y) -> None:
        """Sets the Y coordinate of the lower-right corner of the patch."""
        self.coordinates["lry"] = float(y)

    def load_array(
        self,
        target_edge_size: int = 120,
        target_bands: typing.Union[str, typing.List[str]] = "BGRNIR",
        target_dtype: np.dtype = np.dtype(np.uint16),
        norm_meanstddev: typing.Optional[typing.Tuple[int, int]] = None
    ) -> np.ndarray:
        """Loads the 3-dim array containing the stacked band data for this patch."""
        assert isinstance(target_edge_size, int) and target_edge_size > 0, "invalid patch edge size"
        if isinstance(target_bands, str):
            assert target_bands in sentinel2_band_map, f"invalid target bands code: {target_bands}"
            target_bands = sentinel2_band_map[target_bands]
        assert isinstance(target_bands, list) and len(target_bands) > 0
        assert all([isinstance(b, str) and b in sentinel2_band_names for b in target_bands])
        image = np.zeros((len(target_bands), target_edge_size, target_edge_size), dtype=target_dtype)
        for band_idx, band_name in enumerate(target_bands):
            band_path = self.band_paths_map[band_name]
            assert band_path.suffix == ".tif" and band_path.is_file(), f"invalid file: {band_path}"
            band = cv.imread(str(band_path), flags=cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
            assert band.ndim == 2 and band.shape[0] == band.shape[1] and band.dtype == np.uint16
            if band.shape[0] != target_edge_size:
                band = cv.resize(band, (target_edge_size, target_edge_size), interpolation=cv.INTER_CUBIC)
            if norm_meanstddev is not None:
                assert len(norm_meanstddev) == 2 and all([isinstance(v, float) for v in norm_meanstddev])
                band = (band.astype(np.float) - norm_meanstddev[0]) / norm_meanstddev[1]
            if band.dtype != target_dtype:
                band = band.astype(target_dtype)
            image[band_idx] = band
        return image


class HubCompactor:
    """
    BigEarthNet(-S2) hub dataset repackager/compactor.

    The output of the exportation via this class with be a DIRECTORY of chunks prepared by Hub;
    see https://docs.activeloop.ai/ for more information.

    Note: Adapted from the HDF5 repackaging code of github.com/plstcharles/thelper.
    """

    def __init__(
        self,
        root: typing.Union[typing.AnyStr, pathlib.Path],
        split_file: str,
        classes: typing.List[str],
    ):
        """Parses the raw dataset to validate its metadata and prepare it for exportation."""
        root = pathlib.Path(root).absolute()
        split_file = pathlib.Path(split_file).absolute()
        assert root.is_dir(), f"invalid big earth net root directory path ({root})"
        logger.info(f"loading patch metadata from directory: {root}")
        self.patches = self._load_patch_metadata(root, split_file)
        assert len(self.patches) > 0
        tot_samples = len(self.patches)
        logger.info(f"loaded metadata for {tot_samples} patches")
        self.classes = classes
        self.class_dist = self._compute_class_dist(self.patches)
        self.class2idx = {class_:idx for idx, class_ in enumerate(self.classes)}
        self.idx2class = {idx:class_ for idx, class_ in enumerate(self.classes)}
        self.class_weights = {cname: len(cidxs) / tot_samples for cname, cidxs in self.class_dist.items()}
        logger.debug(f"class weights:\n{pprint.PrettyPrinter(indent=2).pformat(self.class_weights)}")

    @staticmethod
    def _load_patch_metadata(
        root: pathlib.Path,
        split_file: pathlib.Path,
        show_progress_bar: bool = True,
    ) -> typing.List[BigEarthNetPatch]:
        assert root.is_dir(), f"invalid big earth net root directory path ({root})"
        patches = []
        patch_folders = pd.read_csv(split_file, header=None)[0].to_list()  # Each row in the csv is a folder name
        patch_folders = [pathlib.Path(os.path.join(root, f)) for f in patch_folders]
        if show_progress_bar:
            patch_folders = tqdm.tqdm(patch_folders, desc="scanning patch folders")
        for patch_folder in patch_folders:
            match_res = re.match(bigearthnet_patch_name_pattern, patch_folder.name)
            if patch_folder.is_dir():
                band_files_map = {
                    band_name: patch_folder / (patch_folder.name + f"_{band_name}.tif")
                    for band_name in sentinel2_band_names
                }
                assert all([f.is_file() for f in band_files_map.values()])
                metadata_file = patch_folder / (patch_folder.name + "_labels_metadata.json")
                assert metadata_file.is_file(), "unexpected (missing) patch metadata file!"
                with open(metadata_file, "r") as fd:
                    patch_metadata = json.load(fd)
                assert all([k in patch_metadata for k in bigearthnet_metadata_field_names])
                patch_metadata = {k: patch_metadata[k] for k in bigearthnet_metadata_field_names}
                assert len(patch_metadata["labels"]) > 0, "patches should have at least one label?"
                patch_metadata["coordinates"] = {  # convert all coords to float right away
                    k: float(v) for k, v in patch_metadata["coordinates"].items()
                }
                patch_metadata["patch_folder"] = patch_folder.name
                patches.append(BigEarthNetPatch(
                    mission_id=match_res.group(1),
                    tile_col=int(match_res.group(3)),
                    tile_row=int(match_res.group(4)),
                    band_paths_map=band_files_map,
                    **patch_metadata,
                ))
        return patches

    @staticmethod
    def _compute_class_dist(patches: typing.List[BigEarthNetPatch]):
        assert len(patches) > 0
        class_dist = {}
        for idx, patch in enumerate(patches):
            for class_name in patch.labels:
                if class_name not in class_dist:
                    class_dist[class_name] = []
                class_dist[class_name].append(idx)
        return class_dist

    def export(
        self,
        output_path: typing.Union[typing.AnyStr, pathlib.Path],
        target_edge_size: int = 120,
        target_bands: typing.Union[str, typing.List[str]] = "BGR",
        target_dtype: np.dtype = np.dtype(np.uint16),
        norm_meanstddev: typing.Optional[typing.Tuple[int, int]] = None,
        image_compression: str = "lz4",
        show_progress_bar: bool = True,
        **extra_hub_kwargs,
    ) -> None:
        """Exports the dataset to a hub file at the specified location.

        If a hub dataset already exists in the specified location, an exception will be thrown.

        Note that with hub, the output path can be a local path or a remote (server) path under the
        form `PROTOCOL://SERVERNAME/DATASETNAME`. Hub will take care of exporting the data during
        the dataset creation.
        """
        assert pathlib.Path(output_path).suffix == "", "path suffix should be empty (it's a dir!)"
        dataset = hub.empty(str(output_path), overwrite=True, **extra_hub_kwargs)
        with dataset:
            # first, export dataset-level metadata using the info attribute...
            dataset.info.update(
                name="BigEarthNet-S2",
                target_edge_size=target_edge_size,
                target_bands=target_bands,
                norm_meanstddev=norm_meanstddev,
                image_compression=image_compression,
                class_names=self.classes,
            )
            # next, create the tensor fields we'll be filling with our patch data (data + labels)
            dataset.create_tensor(
                name="data",
                htype="image",
                dtype=target_dtype,
                sample_compression=image_compression,
            )
            dataset.create_tensor(
                name="labels",
                htype="sequence[class_label]",
                class_names=self.classes,
            )
            # now, time to actually put the patches inside the dataset...
            patch_iter = list(self.patches)
            if show_progress_bar:
                patch_iter = tqdm.tqdm(patch_iter, desc="exporting patch data")
            for sample_idx, patch in enumerate(patch_iter):
                patch_data = patch.load_array(
                    target_edge_size=target_edge_size,
                    target_bands=target_bands,
                    target_dtype=target_dtype,
                    norm_meanstddev=norm_meanstddev,
                )
                dataset["data"].append(patch_data)
                patch_class_idxs = [self.class2idx[class_name] for class_name in patch.labels]
                try:
                    dataset["labels"].append(patch_class_idxs)
                except Exception as e:
                    print(f"patch_class_idxs = {patch_class_idxs}")
        logger.info(f"Dataset export complete: {str(output_path)}")


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.NOTSET)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root-path",
        help="Path to bigearthnet dataset.",
    )
    parser.add_argument(
        "--output-hub-path",
        help="Path to save constructed hub dataset.",
    )
    parser.add_argument(
        "--splits-path",
        help="Path to splits folder contraining train.csv, val.csv, and test.csv",
    )
    args = parser.parse_args()

    root_path = args.root_path
    output_hub_path = args.output_hub_path
    splits_path = args.splits_path

    if not os.path.isdir(splits_path):
        print(f"Downloading splits to {splits_path}")
        download_full_splits(splits_path)

    dirpath = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(dirpath, "class_list.json"), 'r') as f:
        classes = json.load(f)
    for split in ["train", "val", "test"]:
        dataset = HubCompactor(
            root_path,
            split_file=os.path.join(splits_path, split + ".csv"),
            classes=classes,
        )
        dataset.export(
            os.path.join(output_hub_path, split),
        )
    print("all done")
