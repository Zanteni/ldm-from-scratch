import torch.nn as nn

from nn.normalization import GroupNorm
from nn.attention import MultiHeadAttention


class AttentionBlock(nn.Module):
    """Wraps MultiHeadAttention for spatial (image) feature maps.
    MultiHeadAttention already handles the (B,C,H,W) <-> (B,N,C) reshape
    internally, so this wrapper just needs to norm, attend, and add the
    residual -- no manual reshaping here."""

    def __init__(self, channels, num_groups=32, num_heads=4):
        super().__init__()
        self.norm = GroupNorm(num_groups, channels)
        self.attn = MultiHeadAttention(channels, num_heads=num_heads)

    def forward(self, x):
        h = self.norm(x)
        h = self.attn(h)
        return x + h