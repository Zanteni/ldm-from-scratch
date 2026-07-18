import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.linear import Linear


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding, as used in DDPM (Ho et al. 2020), borrowed
    directly from the Transformer positional embedding (Vaswani et al. 2017).
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "Embedding dimension must be even"
        self.dim = dim
        self.base = base
        self.half_dim = dim // 2
        freq = torch.exp(
            -math.log(self.base)
            * torch.arange(self.half_dim, dtype=torch.float32)
            / self.half_dim
        )
        self.register_buffer("freq", freq)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.float()
        args = t[:, None] * self.freq[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        return emb


class TimestepMLP(nn.Module):
    """
    Projects the fixed sinusoidal timestep embedding through a small
    learnable MLP (Linear -> SiLU -> Linear).
    """

    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.sinusoidal = SinusoidalTimeEmbedding(dim=embedding_dim)
        self.linear1 = Linear(in_features=embedding_dim, out_features=hidden_dim)
        self.linear2 = Linear(in_features=hidden_dim, out_features=out_dim)

    def forward(self, t):
        t = self.sinusoidal(t)
        t = self.linear1(t)
        t = F.silu(t)
        t = self.linear2(t)
        return t