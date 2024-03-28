import math
from typing import Tuple
import zarr
import numpy as np

import gunpowder as gp
from torch.utils.data import IterableDataset
from scipy.ndimage import gaussian_filter
import random

from .load_dataset import download_dataset, get_dataset_path
from .utils.zarr_utils import get_voxel_size
from autoseg.transforms.gp_parser import GunpowderParser


def calc_max_padding(output_size, voxel_size, sigma, mode="shrink"):
    method_padding = gp.Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = gp.Roi(
        (gp.Coordinate([i / 2 for i in [output_size[0], diag, diag]]) + method_padding),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


class GunpowderZarrDataset(IterableDataset):
    def __init__(
        self,
        config,
        input_image_shape: Tuple[int, int, int],
        output_image_shape: Tuple[int, int, int],
    ):
        self.input_image_shape = input_image_shape
        self.output_image_shape = output_image_shape
        self.config = config

        self.gp_parser = GunpowderParser(config)
        self.pipeline = self.gp_parser.parse_config()
        self.voxel_size = gp.Coordinate(self.gp_parser.voxel_size)

    def __iter__(self):
        return iter(self.request_batch(self.input_image_shape, self.output_image_shape))

    def _make_request(self, input_image_shape, output_image_shape):
        input_image_size = gp.Coordinate(input_image_shape) * self.voxel_size
        output_image_size = gp.Coordinate(output_image_shape) * self.voxel_size

        array_keys = self.gp_parser.output_array_keys
        request = gp.BatchRequest()
        for ak in array_keys:
            if ak.identifier == "RAW":
                request.add(ak, gp.Coordinate(input_image_size))
            else:
                request.add(ak, gp.Coordinate(output_image_size))
        return request

    def request_batch(self, input_image_shape, output_image_shape):
        request = self._make_request(input_image_shape, output_image_shape)
        array_keys = self.gp_parser.output_array_keys

        with gp.build(self.pipeline):
            while True:
                sample = self.pipeline.request_batch(request)
                yield tuple(sample[array_keys[i]].data for i in range(len(array_keys)))
