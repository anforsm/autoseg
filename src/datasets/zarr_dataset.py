import math
from typing import Tuple
import zarr
import numpy as np

import gunpowder as gp
from torch.utils.data import IterableDataset
from scipy.ndimage import gaussian_filter
import random


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


class ZarrDataset(IterableDataset):
    def __init__(
        self,
        container_path: str,
        dataset_name,
        num_spatial_dims: int,
        crop_shape: Tuple[int, ...],
        control_point_spacing: int = 100,
        control_point_jitter: float = 100.0,
    ):
        self.container_path = container_path
        self.dataset_name = dataset_name
        self.crop_shape = crop_shape
        self.control_point_spacing = control_point_spacing
        self.control_point_jitter = control_point_jitter
        self.num_spatial_dims = num_spatial_dims
        self.num_dims = len(crop_shape)
        self.num_channels = 3

        assert len(crop_shape) == self.num_spatial_dims, (
            f'"crop_size" must have the same dimension as the '
            f'spatial(temporal) dimensions of the "{self.dataset_config.dataset_name}" '
            f"dataset which is {self.num_spatial_dims}, but it is {crop_shape}"
        )

        with gp.ext.ZarrFile(self.container_path, "r") as z:
            self.shape = z[self.dataset_name].shape
            self.voxel_size = gp.Coordinate(z[self.dataset_name].attrs["resolution"])
        self.crop_size = gp.Coordinate(self.crop_shape) * self.voxel_size
        # self.shape = (100, 100, 100)

        self.__setup_pipeline()

    def __iter__(self):
        return iter(self.__yield_sample())

    def __setup_pipeline(self):
        self.raw = gp.ArrayKey("RAW")
        self.labels = gp.ArrayKey("LABELS")
        self.labels_mask = gp.ArrayKey("LABELS_MASK")
        self.gt_affs = gp.ArrayKey("GT_AFFS")
        self.gt_affs_mask = gp.ArrayKey("GT_AFFS_MASK")
        self.affs_weights = gp.ArrayKey("AFFS_WEIGHTS")

        # treat all dimensions as spatial, with a voxel size of 1
        raw_spec = gp.ArraySpec(interpolatable=True)
        label_spec = gp.ArraySpec(interpolatable=False)

        self.augmentations = gp.NoiseAugment(self.raw) + gp.SimpleAugment(
            transpose_only=[1, 2]
        )

        self.pipeline = gp.ZarrSource(
            self.container_path,
            {
                self.raw: self.dataset_name,
                self.labels: "labels/s0",
                self.labels_mask: "labels_mask/s0",
            },
            {
                self.raw: raw_spec,
                self.labels: label_spec,
                self.labels_mask: label_spec,
            },
        ) + gp.Normalize(self.raw)

        # self.pipeline =
        # gp.AddAffinities(
        #    affinity_neighborhood=[
        #        [-1,0,0],
        #        [0,-1,0],
        #        [0,0,-1]
        #    ],
        #    labels=self.labels,
        #    labels_mask=self.labels_mask,
        #    affinities=self.gt_affs,
        #    affinities_mask=self.gt_affs_mask,
        # )

        # self.random_pipeline = self.pipeline + gp.RandomLocation(mask=self.labels_mask, min_masked=0.1)# + self.augmentations
        # self.random_pipeline = self.pipeline + gp.RandomLocation()# + self.augmentations
        self.random_pipeline = (
            self.pipeline
            + gp.RandomLocation(mask=self.labels_mask, min_masked=0.5)
            + self.augmentations
            + gp.AddAffinities(
                affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
                labels=self.labels,
                labels_mask=self.labels_mask,
                affinities=self.gt_affs,
                affinities_mask=self.gt_affs_mask,
            )
        )

    def request_batch(self, input_shape, output_shape):
        input_size = gp.Coordinate(input_shape) * self.voxel_size
        output_size = gp.Coordinate(output_shape) * self.voxel_size

        labels_padding = calc_max_padding(output_size, self.voxel_size, sigma=40)

        pipeline = self.pipeline
        pipeline += gp.Pad(self.raw, None)
        pipeline += gp.Pad(self.labels, labels_padding)
        pipeline += gp.Pad(self.labels_mask, labels_padding)
        pipeline += gp.RandomLocation(mask=self.labels_mask, min_masked=0.1)
        pipeline += gp.ElasticAugment(
            control_point_spacing=(1, self.voxel_size[0], self.voxel_size[0]),
            jitter_sigma=(0, 5, 5),
            scale_interval=(0.5, 2.0),
            rotation_interval=(0, math.pi / 2),
            subsample=4,
        )
        pipeline += gp.SimpleAugment(transpose_only=[1, 2])
        pipeline += gp.NoiseAugment(self.raw)
        pipeline += gp.IntensityAugment(self.raw, 0.9, 1.1, -0.1, 0.1)
        pipeline += SmoothArray(self.raw)
        pipeline += gp.GrowBoundary(self.labels, only_xy=True)
        pipeline += gp.AddAffinities(
            affinity_neighborhood=[[-1, 0, 0], [0, -1, 0], [0, 0, -1]],
            labels=self.labels,
            labels_mask=self.labels_mask,
            affinities=self.gt_affs,
            affinities_mask=self.gt_affs_mask,
        )
        pipeline += gp.BalanceLabels(self.gt_affs, self.affs_weights, self.gt_affs_mask)
        pipeline += gp.IntensityScaleShift(self.raw, 2, -1)
        pipeline += gp.Unsqueeze([self.raw])
        pipeline += gp.Stack(12)
        pipeline += gp.PreCache(num_workers=20, cache_size=50)

        # pipeline = self.random_pipeline
        # pipeline += gp.PreCache(num_workers=20, cache_size=50)
        with gp.build(pipeline):
            while True:
                request = gp.BatchRequest()
                request.add(self.raw, input_size)
                request.add(self.labels, output_size)
                request.add(self.labels_mask, output_size)
                request.add(self.gt_affs, output_size)
                request.add(self.gt_affs_mask, output_size)
                request.add(self.affs_weights, output_size)
                sample = pipeline.request_batch(request)
                yield sample[self.raw].data, sample[self.labels].data, sample[
                    self.gt_affs
                ].data, sample[self.affs_weights].data

    def __len__(self):
        shape_sum = sum(self.shape)
        crop_size_sum = sum(self.crop_size)
        return shape_sum - crop_size_sum + 1

    def __getitem__(self, index):
        pipeline = self.pipeline  # + self.augmentations
        with gp.build(pipeline):
            # request one sample, all channels, plus crop dimensions
            request = gp.BatchRequest()
            assert len(index) == len(self.crop_size)
            print(index)

            request[self.raw] = gp.ArraySpec(roi=gp.Roi(index, self.crop_size))
            request[self.labels] = gp.ArraySpec(roi=gp.Roi(index, self.crop_size))

            sample = pipeline.request_batch(request)
            return sample[self.raw].data[0], sample[self.labels].data[0]

    def __yield_sample(self):
        """An infinite generator of crops."""

        pipeline = self.random_pipeline
        with gp.build(pipeline):
            while True:
                # request one sample, all channels, plus crop dimensions
                request = gp.BatchRequest()
                # request.add(self.raw, self.crop_size)
                # request.add(self.labels, self.crop_size)
                request[self.raw] = gp.ArraySpec(
                    roi=gp.Roi((0,) * self.num_dims, self.crop_size)
                )
                request[self.labels] = gp.ArraySpec(
                    roi=gp.Roi((0,) * self.num_dims, self.crop_size)
                )
                request[self.labels_mask] = gp.ArraySpec(
                    roi=gp.Roi((0,) * self.num_dims, self.crop_size)
                )

                request[self.gt_affs] = gp.ArraySpec(
                    roi=gp.Roi((0,) * self.num_dims, self.crop_size)
                )

                request[self.gt_affs_mask] = gp.ArraySpec(
                    roi=gp.Roi((0,) * self.num_dims, self.crop_size)
                )

                sample = pipeline.request_batch(request)
                yield sample[self.raw].data, sample[self.labels].data, sample[
                    self.gt_affs
                ].data
