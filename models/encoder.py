import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.conv import Conv2d
from nn.normalization import GroupNorm
from models.resblock import ResBlock, Downsample
from models.attention_block import AttentionBlock


class Encoder(nn.Module):
    """
    (B, 3, 32, 32) -> (B, 2*z_channels, 8, 8)
    Output is raw (mu, logvar) concatenated along channels -- splitting
    happens later in models/vae.py, not here.
    """

    def __init__(self, in_channels=3, base_channels=128, z_channels=4, num_groups=32):
        super().__init__()

        self.conv_in = Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1)

        # Stage 1: 32x32 -> 16x16, channels stay at base_channels (128)
        self.stage1_res1 = ResBlock(base_channels, base_channels, num_groups)
        self.stage1_res2 = ResBlock(base_channels, base_channels, num_groups)
        self.stage1_downsample = Downsample(base_channels)

        # Stage 2: 16x16 -> 8x8, channels double (128 -> 256)
        mid_channels = base_channels * 2
        self.stage2_res1 = ResBlock(base_channels, mid_channels, num_groups)
        self.stage2_res2 = ResBlock(mid_channels, mid_channels, num_groups)
        self.stage2_downsample = Downsample(mid_channels)

        # Middle block: stays at 8x8, mid_channels throughout
        self.mid_res1 = ResBlock(mid_channels, mid_channels, num_groups)
        self.mid_attn = AttentionBlock(mid_channels, num_groups)
        self.mid_res2 = ResBlock(mid_channels, mid_channels, num_groups)

        # Output projection: mid_channels -> 2*z_channels (mu + logvar)
        self.norm_out = GroupNorm(num_groups, mid_channels)
        self.conv_out = Conv2d(mid_channels, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        h = self.conv_in(x)

        h = self.stage1_res1(h)
        h = self.stage1_res2(h)
        h = self.stage1_downsample(h)

        h = self.stage2_res1(h)
        h = self.stage2_res2(h)
        h = self.stage2_downsample(h)

        h = self.mid_res1(h)
        h = self.mid_attn(h)
        h = self.mid_res2(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h  # (B, 2*z_channels, 8, 8)