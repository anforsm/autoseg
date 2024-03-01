from ..config import read_config
from .zarr_dataset import GunpowderZarrDataset


class Kh2015(GunpowderZarrDataset):
    def __init__(self, transform=None):
        if transform is None:
            transform = read_config("examples/no_augments")["pipeline"]

        super().__init__(
            dataset="SynapseWeb/kh2015/oblique",
            dataset_name="raw/s0",
            # labels_dataset_name="labels/s0",
            # labels_mask_dataset_name="labels/mask",
            transform=transform,
            num_spatial_dims=3,
        )
