import zarr
import json


def get_voxel_size(path, ds):
    zr = zarr.open(path, "r")
    zr = zr[ds]
    return zr.attrs["resolution"]


def get_shape(path, ds):
    zr = zarr.open(path, "r")
    zr = zr[ds]
    return zr.shape
