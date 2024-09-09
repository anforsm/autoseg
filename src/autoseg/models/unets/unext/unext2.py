import math
import torch
import torch.nn as nn
from .unext import UNeXt as UNeXtBase
from functools import partial

"""Complete ConvNeXt layer except for kernel size 5 and no residual"""


class ConvPass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels * 4
        self.conv1 = nn.Conv3d(in_channels, in_channels, 5, groups=in_channels)
        self.norm1 = nn.LayerNorm(in_channels, eps=1e-6)
        self.conv2 = nn.Conv3d(in_channels, mid_channels, 1)
        self.act2 = nn.GELU()
        self.conv3 = nn.Conv3d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 4, 1)
        x = self.norm1(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x


UNeXt = partial(
    UNeXtBase,
    ConvPass=ConvPass,
    # debug=True
)
