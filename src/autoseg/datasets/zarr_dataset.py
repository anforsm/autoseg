import math
from typing import Tuple
import zarr
import numpy as np

import gunpowder as gp
from torch.utils.data import IterableDataset
from scipy.ndimage import gaussian_filter
import random

from .load_dataset import download_dataset, get_dataset_path
from autoseg.transforms.gp_parser import GunpowderParser


def calc_max_padding(output_size, voxel_size, sigma, mode="shrink"):
    method_padding = gp.Coordinate((sigma * 3,) * 3)

    diag = np.sqrt(output_size[1] ** 2 + output_size[2] ** 2)

    max_padding = gp.Roi(
        (gp.Coordinate([i / 2 for i in [output_size[0], diag, diag]]) + method_padding),
        (0,) * 3,
    ).snap_to_grid(voxel_size, mode=mode)

    return max_padding.get_begin()


class SmoothArray(gp.BatchFilter):
    def __init__(self, array):
        self.array = array

    def process(self, batch, request):
        array = batch[self.array].data

        assert len(array.shape) == 3

        # different numbers will simulate noisier or cleaner array
        sigma = random.uniform(0.0, 1.0)

        for z in range(array.shape[0]):
            array_sec = array[z]

            array[z] = np.array(gaussian_filter(array_sec, sigma=sigma)).astype(
                array_sec.dtype
            )

        batch[self.array].data = array


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
        self.voxel_size = gp.Coordinate([50, 2, 2])

    def __iter__(self):
        return iter(self.request_batch(self.input_image_shape, self.output_image_shape))

    def request_batch(self, input_image_shape, output_image_shape):
        input_image_size = gp.Coordinate(input_image_shape) * self.voxel_size
        output_image_size = gp.Coordinate(output_image_shape) * self.voxel_size
        print(self.pipeline)
        with gp.build(self.pipeline):
            while True:
                request = gp.BatchRequest()
                for ak in self.gp_parser.array_keys:
                    if ak.identifier == "RAW":
                        request.add(ak, gp.Coordinate(input_image_size))
                    else:
                        request.add(ak, gp.Coordinate(output_image_size))

                sample = self.pipeline.request_batch(request)
                yield tuple(
                    sample[self.gp_parser.array_keys[i]].data
                    for i in range(len(self.gp_parser.array_keys))
                )
