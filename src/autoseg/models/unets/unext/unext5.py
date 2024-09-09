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
from functools import partial

"""ConvNext but with kernel 5, also includes residual connection. No norm. 3x Deeper than previous versions"""


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
            self.act1 = nn.GELU()

            self.conv2 = nn.Conv3d(in_channels, mid_channels, 1)
            self.act2 = nn.GELU()

            self.conv3 = nn.Conv3d(mid_channels, in_channels, 1)
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
        self.down = nn.Conv3d(
            in_channels, out_channels, kernel_size=(1, 3, 3), stride=(1, 3, 3)
        )

    def forward(self, x):
        return self.down(x)


class UNeXt(UNeXtBase):
    def __init__(self, **kwargs):
        print(kwargs)
        super().__init__(**kwargs)
        in_channels = kwargs["in_channels"]
        num_fmaps = kwargs["num_fmaps"]
        fmap_inc_factor = kwargs["fmap_inc_factor"]
        print("unext5")

        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, num_fmaps, 1),
            *[ConvPass(num_fmaps, encoder=True, i=i) for i in range(3)],
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
                for i in range(3)
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
            # nn.Conv3d(num_fmaps*fmap_inc_factor**2, num_fmaps*fmap_inc_factor**3, 1)
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
