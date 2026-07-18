import torch
import torch.nn as nn

from nn.conv import Conv2d
from models.embeddings import TimestepMLP
from models.latent_unet_block import LatentUNetBlock


class LatentUNet(nn.Module):
    """
    Full latent diffusion UNet: predicts the noise added to a latent z_t
    at a given diffusion timestep. Unlike a traditional UNet, there is no
    down/up-sampling of spatial resolution here -- the latent is already
    small (8x8) and heavily compressed, so the whole network operates at
    constant resolution. Depth comes from repeatedly applying the SAME
    LatentUNetBlock instance L times (shared weights, Universal-Transformer-
    style), rather than stacking L independently-weighted blocks.
    """

    def __init__(self, z_channels=4, base_channels=128, time_emb_dim=512,
                 N=6, L=4, max_mult=4, num_groups=32, num_heads=4):
        super().__init__()
        self.L = L
        self.base_channels = base_channels
        self.conv_in = Conv2d(z_channels,base_channels,kernel_size=3,stride=1,padding=1)
        self.time_mlp = TimestepMLP(embedding_dim=base_channels,hidden_dim=time_emb_dim,out_dim=time_emb_dim)

        self.shared_block = LatentUNetBlock(base_channels=base_channels,
                                            time_emb_dim=time_emb_dim,
                                            N=N,
                                            num_groups=num_groups,
                                            num_heads=num_heads,max_mult=max_mult)
        self.conv_out = Conv2d(base_channels,z_channels,kernel_size=3,stride=1,padding=1)
        

    def forward(self, z_t, t):
        t_emb = self.time_mlp(t)
        x = self.conv_in(z_t)
        for l in range(self.L):
            x = self.shared_block(x,t_emb)
        eps_pred = self.conv_out(x)
        return eps_pred