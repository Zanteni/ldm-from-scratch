import torch
import torch.nn as nn

from models.encoder import Encoder
from models.decoder import Decoder


class VAE(nn.Module):
    def __init__(self, in_channels=3, base_channels=128, z_channels=4, num_groups=32):
        super().__init__()
        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            z_channels=z_channels,
            num_groups=num_groups,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            base_channels=base_channels,
            z_channels=z_channels,
            num_groups=num_groups,
        )

    def encode(self, x: torch.Tensor):
        assert x.ndim in (3, 4), f"Expected 3D or 4D tensor, got {x.ndim}D."
        if x.ndim == 3:
            x = x.unsqueeze(0).contiguous()

        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        recon = self.decode(z)
        kl = self.kl_divergence(mu, logvar)

        out = {
            "recon": recon,
            "mu": mu,
            "logvar": logvar,
            "kl": kl,
        }
        return out
    @staticmethod
    def reparametrize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(mu, logvar):
        # sum over channel/spatial dims, mean over batch
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=[1, 2, 3])
        return kl.mean()
    
