import torch
from .unets import (
    ConvPass,
    UNeXt1,
    UNeXt2,
    UNeXt3,
    UNeXt4,
    UNeXt5,
    UNeXt6,
    UNeXt7,
    UNeXt8,
    UNeXt9,
    UNeXt10,
    UNeXt11,
    UNeXt11_2,
    UNeXt11_3,
)


class ConfigurableUNeXt(torch.nn.Module):
    def __init__(
        self,
        # UNet Decoer Params
        in_channels=1,
        output_shapes=[3],
        version="1.0",
        # num_fmaps=12,
        # fmap_inc_factor=5,
        # downsample_factors=((1, 2, 2), (1, 2, 2), (2, 2, 2)),
        # kernel_size_down=(
        #    ((3, 3, 3), (3, 3, 3)),
        #    ((3, 3, 3), (3, 3, 3)),
        #    ((3, 3, 3), (3, 3, 3)),
        #    ((1, 3, 3), (1, 3, 3)),
        # ),
        # kernel_size_up=(
        #    ((3, 3, 3), (3, 3, 3)),
        #    ((3, 3, 3), (3, 3, 3)),
        #    ((3, 3, 3), (3, 3, 3)),
        # ),
        # activation="ReLU",
        # normalization=None,
        **kwargs
    ):
        print("Output shapes", output_shapes)
        super().__init__()

        num_fmaps = 12

        if version == "1.0":
            unet = UNeXt1
        elif version == "2.0":
            unet = UNeXt2
        elif version == "3.0":
            unet = UNeXt3
        elif version == "4.0":
            unet = UNeXt4
        elif version == "5.0":
            unet = UNeXt5
        elif version == "6.0":
            unet = UNeXt6
        elif version == "7.0":
            unet = UNeXt7
        elif version == "7.0":
            unet = UNeXt7
        elif version == "8.0":
            unet = UNeXt8
        elif version == "9.0":
            unet = UNeXt9
        elif version == "10.0":
            unet = UNeXt10
        elif version == "11.0":
            unet = UNeXt11
        elif version == "11.2":
            unet = UNeXt11_2
        elif version == "11.3":
            unet = UNeXt11_3

        print(unet)

        self.unet = unet(
            in_channels=in_channels, num_fmaps=num_fmaps, fmap_inc_factor=5
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
