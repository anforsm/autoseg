import torch
from .unets import UNet, ConvPass, UNETR


class ConfigurableUNETR(torch.nn.Module):
    def __init__(
        self,
        # UNet Decoer Params
        in_channels=1,
        output_shapes=[3],
        num_fmaps=12,
        # Transformer Encoder Params
        image_shape=(128, 128, 128),
        embed_dim=768,
        num_heads=12,
        patch_size=16,
        upsample_factors=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        pad_decoder=True,
        # fmap_inc_factor=5,
        # downsample_factors=((1, 2, 2), (1, 2, 2), (2, 2, 2)),
        # kernel_size_down=(
        #     ((3, 3, 3), (3, 3, 3)),
        #     ((3, 3, 3), (3, 3, 3)),
        #     ((3, 3, 3), (3, 3, 3)),
        #     ((1, 3, 3), (1, 3, 3)),
        # ),
        # kernel_size_up=(
        #     ((3, 3, 3), (3, 3, 3)),
        #     ((3, 3, 3), (3, 3, 3)),
        #     ((3, 3, 3), (3, 3, 3)),
        # ),
        # activation="ReLU",
        # normalization=None,
        **kwargs
    ):
        print("Output shapes", output_shapes)
        super().__init__()

        num_fmaps = 12

        self.unet = UNETR(
            img_shape=image_shape,
            input_dim=in_channels,
            output_dim=num_fmaps,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            upsample_factors=upsample_factors,
            pad_decoder=pad_decoder,
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
