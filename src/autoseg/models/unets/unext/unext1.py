import math
import torch
import torch.nn as nn
from .unext import UNeXt as UNeXtBase
from functools import partial

"""Super simple convpass implementation"""


class ConvPass(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3)
        self.act1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        return x


UNeXt = partial(
    UNeXtBase,
    ConvPass=ConvPass,
)
