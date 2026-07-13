import torch
import torch.nn as nn


class GroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        assert num_channels % num_groups == 0, \
    f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        assert x.ndim == 3 or x.ndim == 4,f"the input should be a 3D or 4D.got{x.ndim}."
        if x.ndim == 3:
            B = 1
            C, H, W = x.shape
        else:
            B, C, H, W = x.shape
        assert C == self.num_channels, \
        f"Expected input with {self.num_channels} channels, got {C}"

        # x: (B, C, H, W)
        x = x.reshape(B,self.num_groups,C//self.num_groups,H,W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x_norm = (x-mean)/torch.sqrt(var+self.eps)
        x = x_norm.reshape(B,C,H,W)
        gamma = self.gamma.reshape(1,C,1,1)
        beta = self.beta.reshape(1,C,1,1)
        x = gamma*x+beta
        return x
        
