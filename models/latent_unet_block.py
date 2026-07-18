import torch
import torch.nn as nn

from models.resblock import TimeResBlock
from models.attention_block import AttentionBlock


def get_channel_multipliers(N, max_mult):
    """
    Returns a list of length N+1 describing the channel multiplier at each
    point in the block's internal N TimeResBlocks. Ramps up by doubling
    until max_mult, plateaus at max_mult as long as needed, then mirrors
    back down to 1. Assumes N is large enough to reach max_mult.
    """
    ramp_up = [1]
    while ramp_up[-1] < max_mult:
        ramp_up.append(ramp_up[-1] * 2)

    ramp_down = ramp_up[:-1][::-1]

    total_needed = N + 1
    plateau_length = total_needed - len(ramp_up) - len(ramp_down)

    return ramp_up + [max_mult] * plateau_length + ramp_down


class LatentUNetBlock(nn.Module):
    def __init__(self, base_channels, time_emb_dim, N, max_mult=4, num_groups=32, num_heads=4):
        super().__init__()

        self.base_channels = base_channels
        self.time_emb_dim = time_emb_dim
        self.N = N
        self.max_mult = max_mult
        self.num_groups = num_groups
        self.num_heads = num_heads

        multiplier_list = get_channel_multipliers(N=N, max_mult=max_mult)

        self.time_resblocks = nn.ModuleList()
        for i in range(N):
            res = TimeResBlock(
                in_channels=base_channels * multiplier_list[i],
                out_channels=base_channels * multiplier_list[i + 1],
                time_emb_dim=time_emb_dim,
                num_groups=num_groups,
            )
            self.time_resblocks.append(res)

        # multiplier_list always ends at 1, so the last TimeResBlock's
        # output is base_channels * 1 = base_channels -- matching this
        # AttentionBlock's channel count, and matching the block's own
        # input width, which is what makes this block composable with
        # itself (block(block(x)) requires input/output shapes to match).
        self.attn = AttentionBlock(
            channels=base_channels,
            num_heads=num_heads,
            num_groups=num_groups,
        )

    def forward(self, x, t_emb):
        for blk in self.time_resblocks:
            x = blk(x, t_emb)
        x = self.attn(x)
        return x