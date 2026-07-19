"""
Simplified DDPM training loss (Ho et al., 2020, Eq. 14).

Rather than optimizing the full variational bound, DDPM trains by sampling
a random timestep and noise, computing x_t via the forward process, and
asking the model to predict the noise that was added. This "predict the
noise" objective, reduced to plain MSE, works as well in practice as the
full ELBO and is what nearly every DDPM implementation actually trains on.
"""

import torch.nn.functional as F
from diffusion.forward_process import forward_diffusion_sample


def compute_diffusion_loss(model, x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, class_labels=None):
    x_t, eps = forward_diffusion_sample(
        x_0=x_0,
        t=t,
        sqrt_alphas_cumprod=sqrt_alphas_cumprod,
        sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
    )
    if class_labels is not None:
        eps_pred = model(x_t, t, class_labels)
    else:
        eps_pred = model(x_t, t)
    return F.mse_loss(eps, eps_pred)