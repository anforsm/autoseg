import math
import torch
import torch.nn as nn
from .unext import UNeXt as UNeXtBase
from functools import partial

import math
import torch
import torch.nn as nn
from .unext import UNeXt as UNeXtBase
from .unext import UNetResAdd, UpsampleBase as Upsample
from .unext7 import LayerNorm
from functools import partial

"""UNext 8 but extra act after second 1x1 pw conv. (end of each convpass)"""
DOWNSAMPLE_FACTOR = 2


class ConvPass(nn.Module):
    def __init__(
        self, in_channels, out_channels=None, encoder=False, i=0  # eg 12  # eg 60
    ):
        super().__init__()

        if not encoder and out_channels is None:
            print(
                "ConvPass either needs to be an encoder convpass (in_channels = out_channels) or out_channels need to be provided"
            )

        self.encoder = encoder
        mid_channels = in_channels * 4

        if self.encoder:
            self.conv1 = nn.Conv3d(
                in_channels,
                in_channels,
                5,
                groups=in_channels,
                padding="valid" if i == 2 else "same",
            )
            self.norm1 = LayerNorm(in_channels)
            self.act1 = nn.GELU()

            self.conv2 = nn.Conv3d(in_channels, mid_channels, 1)
            self.norm2 = LayerNorm(mid_channels)
            self.act2 = nn.GELU()

            self.conv3 = nn.Conv3d(mid_channels, in_channels, 1)
            self.norm3 = LayerNorm(in_channels)
            self.act3 = nn.GELU()
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
            x = self.act1(x)

            x = self.conv2(x)
            x = self.act2(x)

            x = self.conv3(x)
            x = self.act3(x)

            x = UNetResAdd(res, x)
            return x
        else:
            x = self.conv1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.act2(x)
            return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels)
        self.down = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR),
            stride=(1, DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR),
        )

    def forward(self, x):
        x = self.norm(x)
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=(1, DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR),
            stride=(1, DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR),
        )

    def forward(self, x):
        return self.up(x)


class UNeXt(UNeXtBase):
    def __init__(self, **kwargs):
        print(kwargs)
        super().__init__(**kwargs, debug=True)
        in_channels = kwargs["in_channels"]
        num_fmaps = kwargs["num_fmaps"]
        fmap_inc_factor = kwargs["fmap_inc_factor"]

        self.down0 = nn.Sequential(
            nn.Conv3d(in_channels, num_fmaps, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            LayerNorm(num_fmaps),
        )

        self.enc1 = nn.Sequential(
            *[ConvPass(num_fmaps, encoder=True, i=i) for i in range(3)]
        )
        self.down1 = Downsample(num_fmaps, num_fmaps * fmap_inc_factor**1)

        self.enc2 = nn.Sequential(
            *[
                ConvPass(num_fmaps * fmap_inc_factor**1, encoder=True, i=i)
                for i in range(3)
            ]
        )
        self.down2 = Downsample(
            num_fmaps * fmap_inc_factor**1, num_fmaps * fmap_inc_factor**2
        )

        self.enc3 = nn.Sequential(
            *[
                ConvPass(num_fmaps * fmap_inc_factor**2, encoder=True, i=i)
                for i in range(9)
            ]
        )
        self.down3 = Downsample(
            num_fmaps * fmap_inc_factor**2, num_fmaps * fmap_inc_factor**3
        )

        self.enc4 = nn.Sequential(
            *[
                ConvPass(num_fmaps * fmap_inc_factor**3, encoder=True, i=i)
                for i in range(3)
            ],
        )

        # DECODER STAYS THE SAME
        self.up1 = Upsample(
            num_fmaps * fmap_inc_factor**3, num_fmaps * fmap_inc_factor**2
        )
        self.uenc1 = ConvPass(
            2 * num_fmaps * fmap_inc_factor**2, num_fmaps * fmap_inc_factor**2
        )

        self.up2 = Upsample(
            num_fmaps * fmap_inc_factor**2, num_fmaps * fmap_inc_factor**1
        )
        self.uenc2 = ConvPass(
            2 * num_fmaps * fmap_inc_factor**1, num_fmaps * fmap_inc_factor**1
        )

        self.up3 = Upsample(num_fmaps * fmap_inc_factor**1, num_fmaps)
        self.uenc3 = ConvPass(2 * num_fmaps, num_fmaps)

    def forward(self, x):
        x = self.down0(x)
        return super().forward(x)
