"""
Beta/alpha schedules for DDPM (Ho et al., 2020).

The forward diffusion process gradually adds Gaussian noise over T
timesteps according to a variance schedule beta_1, ..., beta_T. From these
betas we derive every other quantity needed throughout training and
sampling: alphas, cumulative alpha products, and the posterior variance
used in the reverse process. Precomputing these once and caching them
avoids recomputing this math at every training step.
"""

import torch
import math


def get_beta_schedule(schedule_type, timesteps, beta_start=1e-4, beta_end=0.02):
    """
    Returns a 1D tensor of betas, shape (timesteps,).

    Args:
        schedule_type: "linear" or "cosine"
        timesteps: total number of diffusion steps T
        beta_start, beta_end: only used for the linear schedule
    """
    if schedule_type == "linear":
        return _linear_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule_type == "cosine":
        return _cosine_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")


def _linear_beta_schedule(timesteps, beta_start, beta_end):
    """
    Original DDPM schedule: betas increase linearly, evenly spaced,
    from beta_start to beta_end over T steps.
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def _cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule from Nichol & Dhariwal, "Improved DDPM" (2021).
    Unlike the linear schedule, betas here are NOT evenly spaced — the
    curve is defined via a cosine shape on alpha_bar directly, keeping
    alpha_bar closer to 1 (less noise) for longer near t=0, then
    transitioning faster later. This avoids destroying image structure
    too early, which the paper found improved sample quality.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_bar = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]  # normalize so alpha_bar_0 = 1
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clip(betas, min=0.0001, max=0.9999)


def get_diffusion_constants(betas):
    """
    Given a beta schedule, precompute every derived quantity needed by the
    forward process, loss, and sampler.

    Returns a dict with:
        betas, alphas
        alphas_cumprod:                alpha_bar_t = prod_{s<=t} alpha_s
        alphas_cumprod_prev:           alpha_bar_{t-1} (alpha_bar_{-1} := 1)
        sqrt_alphas_cumprod:           sqrt(alpha_bar_t)     -- used in q(x_t|x_0)
        sqrt_one_minus_alphas_cumprod: sqrt(1 - alpha_bar_t) -- used in q(x_t|x_0)
        posterior_variance:            variance of q(x_{t-1} | x_t, x_0)
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])

    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }