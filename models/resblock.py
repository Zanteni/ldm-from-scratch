import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.conv import Conv2d
from nn.normalization import GroupNorm


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = GroupNorm(num_groups, in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = GroupNorm(num_groups, out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.skip = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)