"""
Forward diffusion process: samples x_t ~ q(x_t | x_0) directly, without
simulating t sequential noising steps. This is the key trick that makes
DDPM training efficient — a closed-form formula lets you jump straight
from a clean image to its noised version at any timestep.

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps,  eps ~ N(0, I)

As t increases, alpha_bar_t shrinks from ~1 toward ~0, smoothly
interpolating x_t from "all signal" to "all noise."
"""

import torch


def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    Args:
        x_0: clean images, shape (B, C, H, W)
        t: timestep per sample, shape (B,) — may differ across the batch
        sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod: precomputed
            schedule constants, shape (T,)

    Returns:
        x_t: noised images at timestep t, shape (B, C, H, W)
        eps: the actual noise added, shape (B, C, H, W) — needed later to
            compute the training loss against the network's predicted noise
    """
    eps = torch.randn_like(x_0)

    B = t.shape[0]
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t].reshape(B, 1, 1, 1)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t].reshape(B, 1, 1, 1)

    x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * eps

    return x_t, eps