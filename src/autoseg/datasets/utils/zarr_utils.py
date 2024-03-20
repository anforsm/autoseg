import zarr
import json


def get_voxel_size(path):
    with open(path + "/.zattrs", "r") as f:
        data = json.load(f)
    return data["resolution"]
