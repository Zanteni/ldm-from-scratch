import math
import torch
import torch.nn as nn

from nn.normalization import GroupNorm
from nn.linear import Linear


class ClassAttentionBlock(nn.Module):
    """
    Self-attention over spatial tokens PLUS one concatenated class token.
    Unlike true cross-attention, this keeps the class token as an equal
    participant in self-attention (fixing the degenerate softmax problem
    a single-key cross-attention would have), then drops the class token
    afterward so spatial shape is preserved for composability with the
    rest of LatentUNetBlock.

    Implements its own Q/K/V attention math directly (rather than reusing
    MultiHeadAttention), since MultiHeadAttention expects raw (B,C,H,W)
    input and handles its own flatten/unflatten internally -- here we
    already have pre-tokenized (B, N+1, C) sequences with no valid H,W
    to reshape back to mid-computation.
    """

    def __init__(self, channels, class_emb_dim, num_groups=32, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1 / math.sqrt(self.head_dim)

        self.norm = GroupNorm(num_groups, channels)
        self.class_proj = Linear(class_emb_dim, channels)

        self.q = Linear(channels, channels)
        self.k = Linear(channels, channels)
        self.v = Linear(channels, channels)
        self.proj = Linear(channels, channels)

    def forward(self, x, c):
        # x: (B, C, H, W), c: (B, class_emb_dim)
        B, C, H, W = x.shape
        N = H * W

        h = self.norm(x)
        h = h.flatten(2).permute(0, 2, 1)  # (B, N, C)

        c_proj = self.class_proj(c)          # (B, C)
        c_proj = c_proj.unsqueeze(1)         # (B, 1, C)

        combined = torch.cat([h, c_proj], dim=1)  # (B, N+1, C)

        q = self.q(combined)
        k = self.k(combined)
        v = self.v(combined)

        M = N + 1
        q = q.reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, M, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v  # (B, num_heads, M, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, M, C)

        out = self.proj(out)

        out = out[:, :N, :]  # drop the class token, keep only spatial tokens
        out = out.permute(0, 2, 1).reshape(B, C, H, W)

        return x + out