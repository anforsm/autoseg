import torch
from .unets import ConvPass, UNeXt, unext_a


class ConfigurableUNeXt(torch.nn.Module):
    def __init__(
        self,
        # UNet Decoer Params
        in_channels=1,
        num_fmaps=12,
        fmap_inc_factor=5,
        patchify=False,
        stage_ratio=[3, 3, 9, 3],
        output_shapes=[3],
        version="Final",
        bottleneck_fmap_inc=4,
        downsample_factor=2,
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
        if version == "Final":
            unet = UNeXt
        else:
            if version.endswith(".0"):
                classname = version.replace(".0", "")
            else:
                classname = version.replace(".", "_")

            print(unext_a)
            print("loading", classname)
            unet = getattr(unext_a, "UNeXt" + classname)

        self.unet = unet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            stage_ratio=stage_ratio,
            patchify=patchify,
            bottleneck_fmap_inc=bottleneck_fmap_inc,
            downsample_factor=downsample_factor,
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
