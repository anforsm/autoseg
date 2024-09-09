import math
import torch
import torch.nn as nn
from .unext import UNeXt as UNeXtBase
from functools import partial

import math
import torch
import torch.nn as nn
from .unext import UNeXt as UNeXtBase
from .unext import UNetResAdd, UNetResConcat, UpsampleBase as Upsample
from functools import partial

"Has GRN module from convnext2. Also a lot less acts and norms. Restructured as well. Probably incorrectly"


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


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        # B, C, Z, X, Y
        Gx = torch.norm(x, p=2, dim=(2, 3, 4), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


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
                # padding="same",
            )
            self.norm1 = LayerNorm(in_channels)

            self.conv2 = nn.Conv3d(in_channels, mid_channels, 1)
            self.act2 = nn.GELU()
            self.grn = GRN(mid_channels)

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
            x = self.grn(x)

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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels)
        self.down = nn.Conv3d(
            in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = LayerNorm(in_channels)
        self.up = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.up(x)


class UNeXt(UNeXtBase):
    def __init__(self, **kwargs):
        print(kwargs)
        super().__init__(**kwargs)
        in_channels = kwargs["in_channels"]
        num_fmaps = kwargs["num_fmaps"]
        fmap_inc_factor = kwargs["fmap_inc_factor"]
        print("unext5")

        self.patchify = nn.Sequential(
            nn.Conv3d(in_channels, num_fmaps, (1, 2, 2), (1, 2, 2)),
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
            2 * num_fmaps * fmap_inc_factor**3, num_fmaps * fmap_inc_factor**3
        )
        self.uenc1 = ConvPass(
            num_fmaps * fmap_inc_factor**3, num_fmaps * fmap_inc_factor**2
        )

        self.up2 = Upsample(
            2 * num_fmaps * fmap_inc_factor**2, num_fmaps * fmap_inc_factor**2
        )
        self.uenc2 = ConvPass(
            num_fmaps * fmap_inc_factor**2, num_fmaps * fmap_inc_factor**1
        )

        self.up3 = Upsample(
            2 * num_fmaps * fmap_inc_factor**1, num_fmaps * fmap_inc_factor**1
        )
        self.uenc3 = ConvPass(num_fmaps * fmap_inc_factor**1, num_fmaps)

        self.uenc4 = ConvPass(2 * num_fmaps, num_fmaps)

    def forward(self, x):
        # print("hello")
        x = self.patchify(x)
        x = self.enc1(x)
        res1 = x
        # print("res1", res1.shape)

        x = self.down1(x)
        x = self.enc2(x)
        res2 = x
        # print("res2", res2.shape)

        x = self.down2(x)
        x = self.enc3(x)
        res3 = x
        # print("res3", res3.shape)

        x = self.down3(x)
        x = self.enc4(x)
        res4 = x
        # print("res4", res4.shape)

        # print("decoding")
        x = UNetResConcat(res4, x)
        x = self.up1(x)
        # print("post up1")
        # print(x.shape)
        x = self.uenc1(x)
        # print("post enc1")
        # print(x.shape)

        x = UNetResConcat(res3, x)
        x = self.up2(x)
        # print("post up2")
        # print(x.shape)
        x = self.uenc2(x)
        # print("post enc2")
        # print(x.shape)

        x = UNetResConcat(res2, x)
        x = self.up3(x)
        # print("post up3")
        # print(x.shape)
        x = self.uenc3(x)
        # print("post enc3")
        # print(x.shape)

        x = UNetResConcat(res1, x)
        x = self.uenc4(x)
        return x
