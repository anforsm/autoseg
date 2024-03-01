from autoseg.config import read_config
from .zarr_dataset import GunpowderZarrDataset


class Kh2015(GunpowderZarrDataset):
    def __init__(self, input_shape, output_shape, transform=None):
        if transform is None:
            transform = read_config("examples/no_augments")["pipeline"]

        super().__init__(
            dataset="SynapseWeb/kh2015/oblique",
            dataset_name="raw/s0",
            # labels_dataset_name="labels/s0",
            # labels_mask_dataset_name="labels/mask",
            transform=transform,
            num_spatial_dims=3,
            input_image_shape=input_shape,
            output_image_shape=output_shape,
        )
