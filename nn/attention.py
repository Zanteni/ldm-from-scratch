import torch
import torch.nn as nn
import math
from nn.linear import Linear


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, num_heads=4, qkv_bias=True):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv_bias = qkv_bias
        self.scale = 1 / math.sqrt(self.head_dim)

        self.q = Linear(channels, channels, bias=qkv_bias)
        self.k = Linear(channels, channels, bias=qkv_bias)
        self.v = Linear(channels, channels, bias=qkv_bias)

        # Output projection: combines the concatenated multi-head outputs
        # back into a single `channels`-sized representation.
        self.proj = Linear(channels, channels, bias=qkv_bias)

    def forward(self, x):
        assert x.ndim == 3 or x.ndim == 4, f"the input should be a 3D or 4D. got {x.ndim}."
        if x.ndim == 3:
            x = x.unsqueeze(0)
        B, C, H, W = x.shape

        N, D = H * W, C

        # --------------------
        # Flatten image into a sequence of tokens
        # (B, C, H, W) -> (B, N, D)
        # --------------------
        x = x.flatten(2).permute(0, 2, 1)

        # --------------------
        # Q, K, V projections
        # --------------------
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # --------------------
        # Split into heads
        # (B, N, D)
        # -> (B, H, N, Dh)
        # --------------------
        q = q.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # --------------------
        # Attention scores
        # (B,H,N,N)
        # --------------------
        attn = (
            q @ k.transpose(-2, -1)
        ) * self.scale

        # --------------------
        # Softmax
        # --------------------
        attn = torch.softmax(
            attn,
            dim=-1
        )

        # --------------------
        # Attention output
        # (B,H,N,Dh)
        # --------------------
        x = attn @ v

        # --------------------
        # Merge heads
        # (B,H,N,Dh)
        # -> (B,N,D)
        # --------------------
        x = x.permute(
            0,
            2,
            1,
            3,
        ).reshape(
            B,
            N,
            D,
        )

        # --------------------
        # Output projection
        # --------------------
        x = self.proj(x)

        # --------------------
        # Reshape back into an image
        # (B, N, D) -> (B, C, H, W)
        # --------------------
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        return x
