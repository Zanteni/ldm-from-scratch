import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.conv import Conv2d
from nn.normalization import GroupNorm

from nn.linear import Linear
from nn.dropout import Dropout


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

 

class TimeResBlock(nn.Module):
    """
    ResBlock with diffusion timestep embedding injected between the two
    conv stages. Structurally identical to ResBlock above, with one
    addition: a per-block time_proj Linear layer that projects the shared
    t_emb down to this block's channel width, then broadcasts it
    additively across every spatial position (the noise level is a global
    property of the whole sample, not something that varies spatially).
    """

    def __init__(self, in_channels, out_channels, time_emb_dim, num_groups=32, dropout_p=0.1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = GroupNorm(num_groups, in_channels)
        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.time_proj = Linear(time_emb_dim, out_channels)

        self.norm2 = GroupNorm(num_groups, out_channels)
        self.dropout = Dropout(dropout_p)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        t = self.time_proj(t_emb)
        t = t[:, :, None, None]
        h = h + t

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)