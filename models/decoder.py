import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.conv import Conv2d
from nn.normalization import GroupNorm
from models.resblock import ResBlock, Upsample
from models.attention_block import AttentionBlock

class Decoder(nn.Module):
    """
    (B, z_channels, 8, 8) -> (B, out_channels, 32, 32)
    Exact mirror of Encoder: same channel counts, resolutions, and
    block structure, but reversed (channels shrink, resolution grows).
    """

    def __init__(self, out_channels=3, base_channels=128, z_channels=4, num_groups=32):
        super().__init__()

        mid_channels = base_channels * 2  # 256, matches encoder's mid_channels

        self.conv_in = Conv2d(z_channels, mid_channels, kernel_size=3, stride=1, padding=1)

        # Middle block: mirrors encoder's middle block exactly
        self.mid_res1 = ResBlock(mid_channels, mid_channels, num_groups)
        self.mid_attn = AttentionBlock(mid_channels, num_groups)
        self.mid_res2 = ResBlock(mid_channels, mid_channels, num_groups)

        # Up block 1: 8x8 -> 16x16, channels stay at mid_channels (256)
        self.up1_res1 = ResBlock(mid_channels, mid_channels, num_groups)
        self.up1_res2 = ResBlock(mid_channels, mid_channels, num_groups)
        self.up1_upsample = Upsample(mid_channels)

        # Up block 2: 16x16 -> 32x32, channels halve (256 -> 128)
        self.up2_res1 = ResBlock(mid_channels, base_channels, num_groups)
        self.up2_res2 = ResBlock(base_channels, base_channels, num_groups)
        self.up2_upsample = Upsample(base_channels)

        # Output projection: base_channels -> out_channels (image space)
        self.norm_out = GroupNorm(num_groups, base_channels)
        self.conv_out = Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z):
        h = self.conv_in(z)

        h = self.mid_res1(h)
        h = self.mid_attn(h)
        h = self.mid_res2(h)

        h = self.up1_res1(h)
        h = self.up1_res2(h)
        h = self.up1_upsample(h)

        h = self.up2_res1(h)
        h = self.up2_res2(h)
        h = self.up2_upsample(h)

        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h  # (B, 3, 32, 32)