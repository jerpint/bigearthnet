import dataclasses
import datetime
import json
import logging
import os
import pickle
import pprint
import re
import typing

import cv2 as cv
import h5py
import numpy as np
import tqdm

logger = logging.getLogger(__name__)

# BigEarthNet Sentinel-2 Level 2A band mapping
# https://gisgeography.com/sentinel-2-bands-combinations/
# Blue = B2, Green = B3, Red = B4, NIR = B8
bgrnir_band_names = ["B02", "B03", "B04", "B08"]  # all should be 10m resolution (120x120)


@dataclasses.dataclass
class BigEarthNetPatch:

    mission_id: str
    coordinates: typing.Dict[str, float]
    tile_source: str
    tile_row: int
    tile_col: int
    acquisition_date: datetime.datetime
    projection: str
    root_path: str
    band_files: typing.List[str]
    labels: typing.List[str]

    @property
    def ulx(self) -> float:
        return self.coordinates["ulx"]

    @ulx.setter
    def ulx(self, x) -> None:
        self.coordinates["ulx"] = x

    @property
    def uly(self) -> float:
        return self.coordinates["uly"]

    @uly.setter
    def uly(self, y) -> None:
        self.coordinates["uly"] = y

    @property
    def lrx(self) -> float:
        return self.coordinates["lrx"]

    @lrx.setter
    def lrx(self, x) -> None:
        self.coordinates["lrx"] = x

    @property
    def lry(self) -> float:
        return self.coordinates["lry"]

    @lry.setter
    def lry(self, y) -> None:
        self.coordinates["lry"] = y

    def load_array(self,
                   target_size: int = 120,
                   target_bands: typing.Union[str, typing.List[str]] = "bgrnir",
                   target_dtype: np.dtype = np.dtype(np.uint16),
                   norm_meanstddev: typing.Optional[typing.Tuple[int, int]] = None
                   ):
        if isinstance(target_bands, str) and target_bands == "bgrnir":
            target_bands = bgrnir_band_names
        assert len(target_bands) > 0
        image = np.zeros((len(target_bands), target_size, target_size), dtype=target_dtype)
        for band_idx, band_suffix in enumerate(target_bands):
            for band_file in self.band_files:
                if band_file.endswith(band_suffix + ".tif"):
                    band_path = os.path.join(self.root_path, band_file)
                    assert os.path.isfile(band_path), f"could not locate band: {band_path}"
                    band = cv.imread(band_path, flags=cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
                    assert band.ndim == 2 and band.shape[0] == band.shape[1] and band.dtype == np.uint16
                    if band.shape[0] != target_size:
                        band = cv.resize(band, (target_size, target_size), interpolation=cv.INTER_CUBIC)
                    if norm_meanstddev is not None and len(norm_meanstddev) > 0:
                        band = (band.astype(np.float) - norm_meanstddev[0]) / norm_meanstddev[1]
                    if band.dtype != target_dtype:
                        band = band.astype(target_dtype)
                    image[band_idx] = band
        return image



#  import torch torch.utils.data.Dataset):
#  class BigEarthNet(Dataset):
#
#      def __init__(self,
#                   hdf5_path: typing.AnyStr,
#                   cache_path: typing.AnyStr = None,
#                   transforms: typing.Any = None,
#                   meta_keys: typing.Optional[typing.List[str]] = None,
#                   use_global_normalization: bool = True,
#                   keep_file_open: bool = False,
#                   ):
#          super().__init__(transforms, deepcopy=False)
#          logger.info(f"reading BigEarthNet data from: {hdf5_path}")
#          self.hdf5_path = hdf5_path
#          metadata_file = "bigearthnet-meta.pkl"
#          self.metadata_cache = os.path.join(cache_path, metadata_file) if cache_path else None
#          with h5py.File(self.hdf5_path, "r") as hdf5:
#              self.target_size = hdf5.attrs["target_size"]
#              self.target_bands = hdf5.attrs["target_bands"]
#              self.target_dtype = np.dtype(hdf5.attrs["target_dtype"])
#              self.norm_meanstddev = hdf5.attrs["norm_meanstddev"]
#              patch_count = hdf5.attrs["patch_count"]
#              metadata_dataset = hdf5["metadata"]
#              self.samples = []
#              if self.metadata_cache is not None and os.path.exists(self.metadata_cache):
#                  logger.debug(f"parsing metadata from cache file: {self.metadata_cache}")
#                  with open(self.metadata_cache, "rb") as cache:
#                      self.samples = pickle.load(cache)
#                  assert len(self.samples) == patch_count, "unexpected metadata sample count"
#              else:
#                  for sample_idx in tqdm.tqdm(range(patch_count), desc="parsing metadata"):
#                      meta_str = thelper.utils.fetch_hdf5_sample(metadata_dataset, sample_idx)
#                      assert meta_str.startswith("BigEarthNetPatch(")
#                      self.samples.append(eval(meta_str))
#                  if self.metadata_cache:
#                      with open(self.metadata_cache, "wb") as cache:
#                          pickle.dump(self.samples, cache)
#          assert len(self.samples) > 0, "could not load any bigearthnet samples"
#          class_names = []
#          for sample in self.samples:
#              class_names.extend(sample.labels)
#          self.class_names, self.class_counts = np.unique(class_names, return_counts=True)
#          class_map_str = pprint.pformat({n: c for n, c in
#                                          zip(self.class_names, self.class_counts)}, indent=2)
#          logger.debug(f"bigearthnet class sample split:\n{class_map_str}")
#          self.meta_keys = meta_keys if meta_keys is not None else []
#          for meta_key in self.meta_keys:
#              assert hasattr(self.samples[0], meta_key), f"sample missing meta key '{meta_key}'"
#          self.task = thelper.tasks.Classification(class_names=self.class_names, input_key="image",
#                                                   label_key="labels", meta_keys=self.meta_keys,
#                                                   multi_label=True)
#          self.use_global_normalization = use_global_normalization
#          self.image_mean = np.asarray([
#              721.2257105159645,  # B02
#              878.5158627345414,  # B03
#              869.6805989741447,  # B04
#              2432.328152023904,  # B08
#          ], dtype=np.float32)
#          self.image_stddev = np.asarray([
#              1465.656553928543,  # B02
#              1359.523897551790,  # B03
#              1452.286444583796,  # B04
#              1702.876207365026,  # B08
#          ], dtype=np.float32)
#          self.hdf5_handle = h5py.File(self.hdf5_path, "r") if keep_file_open else None
#
#      def __len__(self):
#          return len(self.samples)
#
#      def __getitem__(self, idx):
#          if isinstance(idx, slice):
#              return self._getitems(idx)
#          assert idx < len(self.samples), "sample index is out-of-range"
#          if idx < 0:
#              idx = len(self.samples) + idx
#          # we should try to optimize the I/O... keep file handle open somehow, despite threading?
#          if self.hdf5_handle is not None:
#              image = thelper.utils.fetch_hdf5_sample(self.hdf5_handle["imgdata"], idx)
#          else:
#              with h5py.File(self.hdf5_path, "r") as fd:
#                  image = thelper.utils.fetch_hdf5_sample(fd["imgdata"], idx)
#          assert image.shape[0] == 4, "unexpected band count (curr version supports BGRNIR only)"
#          image = np.transpose(image, (1, 2, 0))
#          if self.use_global_normalization:
#              image = (image.astype(np.float32) - self.image_mean) / self.image_stddev
#          labels = np.asarray([label in self.samples[idx].labels for label in self.class_names])
#          sample = {
#              "image": image,
#              "labels": labels.astype(np.int32),
#              **{meta_key: getattr(self.samples[idx], meta_key) for meta_key in self.meta_keys}
#          }
#          if self.transforms:
#              sample = self.transforms(sample)
#          return sample


class HubCompactor:
    """Adapted from: https://github.com/plstcharles/thelper/blob/master/thelper/data/parsers.py."""

    def __init__(self, root: typing.AnyStr):
        assert os.path.isdir(root), f"invalid big earth net root directory path ({root})"
        metadata_cache_path = os.path.join(root, "patches_metadata.pkl")
        if os.path.exists(metadata_cache_path):
            logger.info(f"loading patch metadata from cache: {metadata_cache_path}")
            with open(metadata_cache_path, "rb") as fd:
                self.patches = pickle.load(fd)
        else:
            logger.info(f"loading patch metadata from directory: {os.path.abspath(root)}")
            self.patches = self._load_patch_metadata(root)
            assert len(self.patches) > 0
            with open(metadata_cache_path, "wb") as fd:
                pickle.dump(self.patches, fd)
        tot_samples = len(self.patches)
        logger.info(f"loaded metadata for {tot_samples} patches")
        self.class_map = self._compute_class_map(self.patches)
        self.classes = list(self.class_map.keys())
        self.class2idx = {class_:idx for idx, class_ in enumerate(self.classes)}
        self.idx2class = {idx:class_ for idx, class_ in enumerate(self.classes)}
        self.class_weights = {cname: len(cidxs) / tot_samples for cname, cidxs in self.class_map.items()}
        logger.debug(f"class weights:\n{pprint.PrettyPrinter(indent=2).pformat(self.class_weights)}")

    @staticmethod
    def _load_patch_metadata(root: typing.AnyStr, progress_bar: bool = True):
        assert os.path.isdir(root), f"invalid big earth net root directory path ({root})"
        name_pattern = re.compile(r"^([\w\d]+)_MSIL2A_(\d{8}T\d{6})_(\d+)_(\d+)$")
        patches = []
        patch_folders = os.listdir(root)
        patch_iter = tqdm.tqdm(patch_folders) if progress_bar else patch_folders
        for patch_folder in patch_iter:
            match_res = re.match(name_pattern, patch_folder)
            patch_folder_path = os.path.join(root, patch_folder)
            if match_res and os.path.isdir(patch_folder_path):
                patch_files = os.listdir(patch_folder_path)
                band_files = [p for p in patch_files if p.endswith(".tif")]
                metadata_files = [p for p in patch_files if p.endswith(".json")]
                assert len(band_files) == 12 and len(metadata_files) == 1
                metadata_path = os.path.join(patch_folder_path, metadata_files[0])
                with open(metadata_path, "r") as fd:
                    patch_metadata = json.load(fd)
                expected_meta_keys = ["labels", "coordinates", "projection", "tile_source", "acquisition_date"]
                assert all([key in patch_metadata for key in expected_meta_keys])
                acquisition_timestamp = datetime.datetime.strptime(patch_metadata["acquisition_date"],
                                                                   "%Y-%m-%d %H:%M:%S")
                file_timestamp = datetime.datetime.strptime(match_res.group(2), "%Y%m%dT%H%M%S")
                assert acquisition_timestamp == file_timestamp
                patches.append(BigEarthNetPatch(
                    root_path=os.path.abspath(patch_folder_path),
                    mission_id=match_res.group(1),
                    tile_col=int(match_res.group(3)),
                    tile_row=int(match_res.group(4)),
                    band_files=sorted(band_files),
                    **patch_metadata,
                ))
        return patches

    @staticmethod
    def _compute_class_map(patches: typing.List[BigEarthNetPatch]):
        assert len(patches) > 0
        class_map = {}
        for idx, patch in enumerate(patches):
            for class_name in patch.labels:
                if class_name not in class_map:
                    class_map[class_name] = []
                class_map[class_name].append(idx)
        return class_map

    def export(self,
               output_hub_path: typing.AnyStr,
               target_size: int = 120,
               target_bands: typing.Union[str, typing.List[str]] = "bgrnir",
               target_dtype: np.dtype = np.dtype(np.uint16),
               norm_meanstddev: typing.Optional[typing.Tuple[int, int]] = None,
               metadata_compression: typing.Optional[typing.Any] = None,
               image_compression: typing.Optional[typing.Any] = "chunk_lz4",
               progress_bar: bool = True,
               ):

        import hub

        if isinstance(target_bands, str) and target_bands == "bgrnir":
            target_bands = bgrnir_band_names

        patch_iter = tqdm.tqdm(self.patches, desc="exporting image data")

        ds = hub.empty(output_hub_path) # Create the dataset locally
        with ds:
            # Create the tensors with names of your choice.
            ds.create_tensor("images", dtype=np.uint16, sample_compression=None)
            #  ds.create_tensor('labels', htype = 'class_label', class_names = class_names)
            ds.create_tensor("labels", htype="sequence[class_label]", sample_compression=None) #, class_names=self.classes)

            ds.info.update(description = 'BigEarth Net dataset.')

            for sample_idx, patch in enumerate(patch_iter):
                patch_array = patch.load_array(
                    target_size=target_size,
                    target_bands=target_bands,
                    norm_meanstddev=norm_meanstddev
                )
                assert patch_array.shape == (len(target_bands), target_size, target_size)
                ds.images.append(patch_array)

                labels = [np.uint32(self.class2idx[label]) for label in patch.labels]
                ds.labels.append(labels)

    #  def export_hdf5(self,
    #             output_hdf5_path: typing.AnyStr,
    #             target_size: int = 120,
    #             target_bands: typing.Union[str, typing.List[str]] = "bgrnir",
    #             target_dtype: np.dtype = np.dtype(np.uint16),
    #             norm_meanstddev: typing.Optional[typing.Tuple[int, int]] = None,
    #             metadata_compression: typing.Optional[typing.Any] = None,
    #             image_compression: typing.Optional[typing.Any] = "chunk_lz4",
    #             progress_bar: bool = True,
    #             ):
    #      logger.info(f"exporting BigEarthNet to {output_hdf5_path}")
    #      if isinstance(target_bands, str) and target_bands == "bgrnir":
    #          target_bands = bgrnir_band_names
    #      pretty = pprint.PrettyPrinter(indent=2)
    #      with h5py.File(output_hdf5_path, "w") as fd:
    #          fd.attrs["source"] = thelper.utils.get_log_stamp()
    #          fd.attrs["git_sha1"] = thelper.utils.get_git_stamp()
    #          fd.attrs["version"] = thelper.__version__
    #          fd.attrs["target_size"] = target_size
    #          fd.attrs["target_bands"] = target_bands
    #          fd.attrs["target_dtype"] = target_dtype.str
    #          fd.attrs["norm_meanstddev"] = () if not norm_meanstddev else norm_meanstddev
    #          fd.attrs["metadata_compression"] = pretty.pformat(metadata_compression)
    #          fd.attrs["image_compression"] = pretty.pformat(image_compression)
    #          fd.attrs["patch_count"] = len(self.patches)
    #          logger.debug("dataset attributes: \n" +
    #                       pretty.pformat({key: val for key, val in fd.attrs.items()}))
    #          logger.debug("generating meta packets...")
    #          patch_meta_strs = np.asarray([repr(p) for p in self.patches])
    #          logger.debug("creating datasets...")
    #          assert metadata_compression not in thelper.utils.chunk_compression_flags
    #          metadata = thelper.utils.create_hdf5_dataset(
    #              fd=fd, name="metadata", max_len=len(patch_meta_strs),
    #              batch_like=patch_meta_strs, compression=metadata_compression)
    #          target_tensor_shape = (len(target_bands), target_size, target_size)
    #          fake_batch = np.zeros((1, *target_tensor_shape), dtype=target_dtype)
    #          if image_compression in thelper.utils.chunk_compression_flags or \
    #                  image_compression in thelper.utils.no_compression_flags:
    #              imgdata = thelper.utils.create_hdf5_dataset(
    #                  fd=fd,
    #                  name="imgdata",
    #                  max_len=len(self.patches),
    #                  batch_like=fake_batch,
    #                  compression=image_compression,
    #                  chunk_size=(1, *target_tensor_shape),
    #                  flatten=False
    #              )
    #          else:
    #              imgdata = thelper.utils.create_hdf5_dataset(
    #                  fd=fd,
    #                  name="imgdata",
    #                  max_len=len(self.patches),
    #                  batch_like=fake_batch,
    #                  compression=image_compression,
    #                  chunk_size=None,
    #                  flatten=True
    #              )
    #          logger.debug("exporting metadata...")
    #          if progress_bar:
    #              patch_meta_iter = tqdm.tqdm(patch_meta_strs, desc="exporting metadata")
    #          else:
    #              patch_meta_iter = patch_meta_strs
    #          for sample_idx, patch_meta_str in enumerate(patch_meta_iter):
    #              thelper.utils.fill_hdf5_sample(
    #                  metadata, sample_idx, sample_idx, patch_meta_strs, None)
    #          logger.debug("exporting image data...")
    #          if progress_bar:
    #              patch_iter = tqdm.tqdm(self.patches, desc="exporting image data")
    #          else:
    #              patch_iter = self.patches
    #          for sample_idx, patch in enumerate(patch_iter):
    #              patch_array = patch.load_array(
    #                  target_size=target_size,
    #                  target_bands=target_bands,
    #                  norm_meanstddev=norm_meanstddev
    #              )
    #              assert patch_array.shape == (len(target_bands), target_size, target_size)
    #              thelper.utils.fill_hdf5_sample(
    #                  imgdata, sample_idx, 0, patch_array.reshape((1, *target_tensor_shape)))

# TODO: Replace with hub when ready
#  def _compute_statistics(dset: BigEarthNet) -> typing.Dict:
#      array_alloc_size = len(dset)
#      stat_map = {
#          # alloc three vals per item: px-wise sum, px-wise sqsum, px count
#          band: np.zeros((array_alloc_size, 3), dtype=np.float64)
#          for band in bgrnir_band_names
#      }
#      for batch_idx, sample in enumerate(tqdm.tqdm(dataset)):
#          image = sample["image"]
#          assert image.ndim == 3 and image.shape[-1] == len(bgrnir_band_names)
#          for ch_idx, ch in enumerate(bgrnir_band_names):
#              image_ch = image[..., ch_idx]
#              stat_map[ch][batch_idx] = (
#                  np.sum(image_ch, dtype=np.float64),
#                  np.sum(np.square(image_ch, dtype=np.float64)),
#                  np.float64(image_ch.size),
#              )
#      for key, array in stat_map.items():
#          tot_size = np.sum(array[:, 2])
#          mean = np.sum(array[:, 0]) / tot_size
#          stddev = np.sqrt(np.sum(array[:, 1]) / tot_size - np.square(mean))
#          stat_map[key] = {"mean": mean, "stddev": stddev}
#      return stat_map


if __name__ == "__main__":
    # @@@@ TODO: CONVERT TO PROPER TEST
    logging.basicConfig()
    logging.getLogger().setLevel(logging.NOTSET)
    #  root_path = "/shared/data_ufast_ext4/datasets/bigearthnet/BigEarthNet-v1.0"
    root_path = "debug_dataset/"
    output_hub_path = "hub_dataset/"
    dataset = HubCompactor(root_path)
    dataset.export(output_hub_path)

    #dataset._test_close_vals("/shared/data_sfast/datasets/bigearthnet/bigearthnet-thelper.hdf5")
    #  dataset = BigEarthNet(
    #      hdf5_path="/shared/data_sfast/datasets/bigearthnet/bigearthnet-thelper.hdf5",
    #      cache_path="data/cache",
    #      keep_file_open=True,
    #  )
    #  stat_map = _compute_statistics(dataset)
    #  logging.info(f"stat_map =\n{pprint.pformat(stat_map, indent=4)}")
    #  print("all done")
