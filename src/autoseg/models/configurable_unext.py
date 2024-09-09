import torch
from .unets import ConvPass, unext_a


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

        if version.endswith(".0"):
            classname = version.replace(".0", "")
        else:
            classname = version.replace(".", "_")

        print(unext_a)
        print("loading", classname)
        unet = getattr(unext_a, "UNeXt" + classname)

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
