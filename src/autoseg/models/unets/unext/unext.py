import math
import torch
import torch.nn as nn
from functools import partial


def UNetResAdd(a, b):
    """Center crop a so that it can be added with b
    a: [b, c, z, y, x]
    b: [b, c, z, y, x]
    """
    a_shape = a.shape
    b_shape = b.shape

    # Calculate offsets for center cropping
    z_diff = a_shape[2] - b_shape[2]
    y_diff = a_shape[3] - b_shape[3]
    x_diff = a_shape[4] - b_shape[4]

    z_start = z_diff // 2
    y_start = y_diff // 2
    x_start = x_diff // 2

    # Perform center cropping
    a_cropped = a[
        :,
        :,
        z_start : z_start + b_shape[2],
        y_start : y_start + b_shape[3],
        x_start : x_start + b_shape[4],
    ]

    # Concatenate along the channel dimension
    return a_cropped + b


def UNetResConcat(a, b):
    """Center crop a so that it can be concatenated with b
    a: [b, c, z, y, x]
    b: [b, c, z, y, x]
    """
    a_shape = a.shape
    b_shape = b.shape

    # Calculate offsets for center cropping
    z_diff = a_shape[2] - b_shape[2]
    y_diff = a_shape[3] - b_shape[3]
    x_diff = a_shape[4] - b_shape[4]

    z_start = z_diff // 2
    y_start = y_diff // 2
    x_start = x_diff // 2

    # Perform center cropping
    a_cropped = a[
        :,
        :,
        z_start : z_start + b_shape[2],
        y_start : y_start + b_shape[3],
        x_start : x_start + b_shape[4],
    ]

    # Concatenate along the channel dimension
    return torch.cat([a_cropped, b], dim=1)


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


DOWNSAMPLE_FACTOR = 2


class ConvPass(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        bottleneck_fmap_inc=4,
        encoder=False,
        i=0,  # eg 12  # eg 60
    ):
        super().__init__()

        if not encoder and out_channels is None:
            print(
                "ConvPass either needs to be an encoder convpass (in_channels = out_channels) or out_channels need to be provided"
            )

        self.encoder = encoder
        mid_channels = in_channels * bottleneck_fmap_inc

        if self.encoder:
            self.conv1 = nn.Conv3d(
                in_channels,
                in_channels,
                5,
                groups=in_channels,
                padding="valid" if i == 2 else "same",
            )
            self.norm1 = LayerNorm(in_channels)

            self.conv2 = nn.Conv3d(in_channels, mid_channels, 1)
            self.act2 = nn.GELU()

            self.conv3 = nn.Conv3d(mid_channels, in_channels, 1)
        else:
            self.conv1 = nn.Conv3d(in_channels, out_channels, 3)
            self.act1 = nn.GELU()

            self.conv2 = nn.Conv3d(out_channels, out_channels, 3)
            self.act2 = nn.GELU()

    def forward(self, x):
        if self.encoder:
            res = x
            x = self.conv1(x)
            x = self.norm1(x)

            x = self.conv2(x)
            x = self.act2(x)

            x = self.conv3(x)

            x = UNetResAdd(res, x)
            return x
        else:
            x = self.conv1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.act2(x)
            return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_factor):
        super().__init__()
        self.norm = LayerNorm(in_channels)
        self.down = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, downsample_factor, downsample_factor),
            stride=(1, downsample_factor, downsample_factor),
        )

    def forward(self, x):
        x = self.norm(x)
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_factor):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(1, downsample_factor, downsample_factor),
            stride=(1, downsample_factor, downsample_factor),
        )

    def forward(self, x):
        return self.up(x)


class UNeXt(nn.Module):
    def __init__(
        self,
        in_channels=1,
        num_fmaps=12,
        fmap_inc_factor=5,
        bottleneck_fmap_inc=4,
        stage_ratio=[3, 3, 9, 3],
        patchify=False,
        downsample_factor=2,
    ):
        super().__init__()

        if patchify:
            first = nn.Sequential(
                nn.Conv3d(
                    in_channels, num_fmaps, kernel_size=(1, 2, 2), stride=(1, 2, 2)
                ),
                LayerNorm(num_fmaps),
            )
        else:
            first = nn.Conv3d(
                in_channels, num_fmaps, kernel_size=(1, 1, 1), stride=(1, 1, 1)
            )

        self.enc1 = nn.Sequential(
            first,
            *[
                ConvPass(
                    num_fmaps,
                    bottleneck_fmap_inc=bottleneck_fmap_inc,
                    encoder=True,
                    i=i,
                )
                for i in range(stage_ratio[0])
            ],
        )
        self.down1 = Downsample(
            num_fmaps,
            num_fmaps * fmap_inc_factor**1,
            downsample_factor=downsample_factor,
        )

        self.enc2 = nn.Sequential(
            *[
                ConvPass(
                    num_fmaps * fmap_inc_factor**1,
                    bottleneck_fmap_inc=bottleneck_fmap_inc,
                    encoder=True,
                    i=i,
                )
                for i in range(stage_ratio[1])
            ]
        )
        self.down2 = Downsample(
            num_fmaps * fmap_inc_factor**1,
            num_fmaps * fmap_inc_factor**2,
            downsample_factor=downsample_factor,
        )

        self.enc3 = nn.Sequential(
            *[
                ConvPass(
                    num_fmaps * fmap_inc_factor**2,
                    bottleneck_fmap_inc=bottleneck_fmap_inc,
                    encoder=True,
                    i=i,
                )
                for i in range(stage_ratio[2])
            ]
        )
        self.down3 = Downsample(
            num_fmaps * fmap_inc_factor**2,
            num_fmaps * fmap_inc_factor**3,
            downsample_factor=downsample_factor,
        )

        self.enc4 = nn.Sequential(
            *[
                ConvPass(
                    num_fmaps * fmap_inc_factor**3,
                    bottleneck_fmap_inc=bottleneck_fmap_inc,
                    encoder=True,
                    i=i,
                )
                for i in range(stage_ratio[3])
            ],
        )

        # DECODER STAYS THE SAME
        self.up1 = Upsample(
            num_fmaps * fmap_inc_factor**3,
            num_fmaps * fmap_inc_factor**2,
            downsample_factor=downsample_factor,
        )
        self.uenc1 = ConvPass(
            2 * num_fmaps * fmap_inc_factor**2, num_fmaps * fmap_inc_factor**2
        )

        self.up2 = Upsample(
            num_fmaps * fmap_inc_factor**2,
            num_fmaps * fmap_inc_factor**1,
            downsample_factor=downsample_factor,
        )
        self.uenc2 = ConvPass(
            2 * num_fmaps * fmap_inc_factor**1, num_fmaps * fmap_inc_factor**1
        )

        self.up3 = Upsample(
            num_fmaps * fmap_inc_factor**1,
            num_fmaps,
            downsample_factor=downsample_factor,
        )
        self.uenc3 = ConvPass(2 * num_fmaps, num_fmaps)

    def forward(self, x):
        res1 = self.enc1(x)
        x = self.down1(res1)

        res2 = self.enc2(x)
        x = self.down2(res2)

        res3 = self.enc3(x)
        x = self.down3(res3)

        x = self.enc4(x)

        x = self.up1(x)
        x = UNetResConcat(res3, x)
        x = self.uenc1(x)

        x = self.up2(x)
        x = UNetResConcat(res2, x)
        x = self.uenc2(x)

        x = self.up3(x)
        x = UNetResConcat(res1, x)
        x = self.uenc3(x)

        return x
