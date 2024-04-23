import torch
from .unets import UNet, ConvPass


class ConfigurableUNet(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        output_shapes=[3],
        num_fmaps=12,
        fmap_inc_factor=5,
        num_fmaps_out=12,
        downsample_factors=((1, 2, 2), (1, 2, 2), (2, 2, 2)),
        kernel_size_down=(
            ((3, 3, 3), (3, 3, 3)),
            ((3, 3, 3), (3, 3, 3)),
            ((3, 3, 3), (3, 3, 3)),
            ((1, 3, 3), (1, 3, 3)),
        ),
        kernel_size_up=(
            ((3, 3, 3), (3, 3, 3)),
            ((3, 3, 3), (3, 3, 3)),
            ((3, 3, 3), (3, 3, 3)),
        ),
        activation="ReLU",
        normalization=None,
    ):
        super().__init__()

        self.unet = UNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            constant_upsample=True,
            activation=activation,
            normalization=normalization,
            num_heads=1,
        )

        self.heads = torch.nn.ModuleList(
            [
                ConvPass(
                    num_fmaps,
                    shape,
                    [[1, 1, 1]],
                    activation="Sigmoid",
                    normalization=None,
                )
                for shape in output_shapes
            ]
        )

    def forward(self, input):
        z = self.unet(input)

        return tuple(head(z) for head in self.heads)
        # return self.head(z)
