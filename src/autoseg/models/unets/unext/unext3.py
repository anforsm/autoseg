import math
import torch
import torch.nn as nn
from .unext import UNeXt as UNeXtBase
from functools import partial

"""ConvNext implementetion but with kernel 5 and no norm nor residual. Instead activation after every convolution."""


class ConvPass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels * 4
        self.conv1 = nn.Conv3d(in_channels, in_channels, 5, groups=in_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv3d(in_channels, mid_channels, 1)
        self.act2 = nn.GELU()

        self.conv3 = nn.Conv3d(mid_channels, out_channels, 1)
        self.act3 = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        return x


UNeXt = partial(
    UNeXtBase,
    ConvPass=ConvPass,
    # debug=True
)
