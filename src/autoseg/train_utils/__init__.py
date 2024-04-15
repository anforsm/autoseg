import numpy as np
import torch
from PIL import Image
from einops import rearrange
import zarr
import os


def get_random_color():
    return tuple(np.random.randint(0, 255, 3))


def segmentation_to_rgb(segmentation_arr):
    # Creates an RGB Image with a unique color for each label_id in the segmentation_arr
    # segmentation_arr is of shape (X, Y)
    # segmentation_arr should be a numpy array
    # segmentation_arr should be in the range [0, 255]
    # segmentation_arr should be of type np.uint8
    colors = {}
    rgb_arr = np.zeros(segmentation_arr.shape + (3,), dtype=np.uint8)
    for x in range(segmentation_arr.shape[0]):
        for y in range(segmentation_arr.shape[1]):
            label = segmentation_arr[x, y]
            if label == 0:
                colors[label] = (0, 0, 0)
            if label not in colors:
                colors[label] = get_random_color()
            rgb_arr[x, y] = colors[label]
    return rgb_arr


def get_smallest_volume(volumes, spatial_dims=2):
    smallest_shape = None
    for volume in volumes:
        shape = volume.shape[-spatial_dims:]
        if smallest_shape is None or sum(shape) < sum(smallest_shape):
            smallest_shape = shape
    return smallest_shape


def get_largest_volume(volumes, spatial_dims=2):
    largest_shape = None
    for volume in volumes:
        shape = volume.shape[-spatial_dims:]
        if largest_shape is None or sum(shape) > sum(largest_shape):
            largest_shape = shape
    return largest_shape


def get_2D_snapshot(volumes, center_crop=True):
    # Each volume should be of size (B, C, Z, Y, X)
    # Also accepted is (B, Z, Y, X)
    smallest_shape = get_smallest_volume(volumes.values())

    volumes_2d = []
    for name, volume in volumes.items():
        # Get the middle slice of the 3D volume
        middle_slice = volume.shape[-3] // 2

        # Remove batch dimension
        volume = volume[0]
        if len(volume.shape) == 3:
            # Add channel dimension
            volume = rearrange(volume, "z y x -> () z y x")

        volume = volume[:, middle_slice, :, :]

        # Rearrange the volume to (H, W, C)
        if volume.shape[0] == 1:
            volume = rearrange(volume, "() h w -> h w")
        else:
            volume = rearrange(volume, "c h w -> h w c")

        if center_crop:
            # Center crop the volume
            diff = np.array(volume.shape[:2]) - np.array(smallest_shape)
            offset = diff // 2
            if not (offset[0] == 0 and offset[1] == 0):
                volume = volume[offset[0] : -offset[0], offset[1] : -offset[1]]

        # if volume[0, 0, 0].dtype == np.float32:
        #    volume = (volume * 255).astype(np.uint8)
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().detach().numpy()

        if len(volume.shape) > 2 and volume.shape[2] > 3:
            volume = volume[:, :, :3]

        if name == "affs":
            volume = volume.clip(-1, 1)

        if volume.max() <= 1 and volume.min() >= 0:
            volume = (volume * 255).astype(np.uint8)
        elif volume.max() <= 1 and volume.min() >= -1:
            volume = ((volume + 1) / 2 * 255).astype(np.uint8)

        if volume.dtype == np.uint64 or volume.dtype == np.uint32:
            volume = segmentation_to_rgb(volume)

        # print(f"{name} shape: {volume.shape}, dtype: {volume.dtype}, min: {volume.min()}, max: {volume.max()}")
        try:
            Image.fromarray(volume)
        except Exception as e:
            print(volume)
            print(volume.shape)
            print("Failed to create image for ", name)
            print(e)
            exit()

        volumes_2d.append(volume)

    # create images
    return tuple(Image.fromarray(volume) for volume in volumes_2d)


def save_zarr_snapshot(filename, dataset_prefix, volumes, resolution=[1, 1, 1]):
    os.makedirs(filename, exist_ok=True)
    print(filename)
    # smallest_shape = get_smallest_volume(volumes.values(), 3)
    largest_shape = get_largest_volume(volumes.values(), 3)
    f = zarr.open(filename, "a")

    for name, volume in volumes.items():
        if isinstance(volume, torch.Tensor):
            volume = volume.cpu().detach().numpy()
        # Remove batch dimension
        volume = volume[0]

        if volume.min() < 0 and volume.min() > -1 and volume.max() < 1:
            volume = ((volume + 1) / 2 * 255).astype(np.uint8)

        if "int" in str(volume.dtype) and volume.max() <= 1:
            volume = volume.astype(np.float32)

        f[f"{dataset_prefix}/{name}"] = volume
        f[f"{dataset_prefix}/{name}"].attrs["resolution"] = resolution
        diff = np.array(largest_shape) - np.array(volume.shape)[-3:]
        offset = diff // 2
        f[f"{dataset_prefix}/{name}"].attrs["offset"] = list(offset)

    return filename + "/" + dataset_prefix
