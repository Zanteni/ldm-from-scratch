import torch


def ddim_step(model, x_t, t1, t2, sqrt_alphas_cumprod, alphas_cumprod, device="cpu"):
    """
    Single DDIM step: jumps directly from timestep t1 to timestep t2
    (t2 < t1, possibly skipping many steps), using the model's noise
    prediction at t1. Deterministic (no noise added), unlike DDPM's
    ancestral sampler.
    """
    B = x_t.shape[0]
    t1_B = torch.full((B,), t1, dtype=torch.long, device=device)

    eps_pred = model(x_t, t1_B)

    alpha_bar_t1 = alphas_cumprod[t1]
    alpha_bar_t2 = alphas_cumprod[t2]

    x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t1) * eps_pred) / torch.sqrt(alpha_bar_t1)
    x_t2 = torch.sqrt(alpha_bar_t2) * x0_pred + torch.sqrt(1 - alpha_bar_t2) * eps_pred

    return x_t2

import torch


@torch.no_grad()
def ddim_sample(model, shape, T, num_steps, sqrt_alphas_cumprod, alphas_cumprod, device="cpu"):
    """
    DDIM sampling: generates a clean z_0 from pure noise, using only
    `num_steps` model calls instead of the full T steps.

    Args:
        shape: (B, C, H, W) -- desired output shape
        T: total number of diffusion timesteps the model was trained with
        num_steps: how many steps to actually use for sampling (<= T)
    Returns:
        z_0: final denoised sample, shape (B, C, H, W)
    """
    reduced_schedule = torch.linspace(0, T-1, num_steps).long()
    reversed_reduced_schedule = torch.flip(reduced_schedule,dims=[0])
    x_t = torch.randn(shape, device=device)
    for t,tprev in zip(reversed_reduced_schedule[:-1],reversed_reduced_schedule[1:]):
        x_tprev = ddim_step(model=model,
                            x_t=x_t,
                            t1=t.item(),
                            t2=tprev.item(),
                            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                            alphas_cumprod=alphas_cumprod,
                            device=device)
        x_t = x_tprev
    return x_t

def cfg_ddim_step(model, x_t, t1, t2, class_labels, guidance_scale, sqrt_alphas_cumprod, alphas_cumprod, num_classes, device="cpu"):
    B = x_t.shape[0]
    t1_B = torch.full((B,), t1, dtype=torch.long, device=device)

    # conditional prediction
    eps_cond = model(x_t, t1_B, class_labels)

    # unconditional prediction -- same model, but with the null-class index
    null_labels = torch.full((B,), num_classes, dtype=torch.long, device=device)
    eps_uncond = model(x_t, t1_B, null_labels)

    # combine: push away from unconditional, toward conditional
    eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

    # rest is identical to the unconditional ddim_step, just using eps_pred
    alpha_bar_t1 = alphas_cumprod[t1]
    alpha_bar_t2 = alphas_cumprod[t2]

    x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t1) * eps_pred) / torch.sqrt(alpha_bar_t1)
    x_t2 = torch.sqrt(alpha_bar_t2) * x0_pred + torch.sqrt(1 - alpha_bar_t2) * eps_pred

    return x_t2

@torch.no_grad()
def cfg_ddim_sample(model, shape, T, num_steps, class_labels, guidance_scale, sqrt_alphas_cumprod, alphas_cumprod, num_classes, device="cpu"):
    """
    CFG-guided DDIM sampling: same structure as ddim_sample, but each step
    uses cfg_ddim_step (two model calls -- conditional and unconditional --
    blended by guidance_scale) instead of a single unconditional call.
    """
    reduced_schedule = torch.linspace(0, T-1, num_steps).long()
    reversed_reduced_schedule = torch.flip(reduced_schedule,dims=[0])
    x_t = torch.randn(shape, device=device)
    for t,tprev in zip(reversed_reduced_schedule[:-1],reversed_reduced_schedule[1:]):
        x_tprev = cfg_ddim_step(model=model,
                            x_t=x_t,
                            t1=t.item(),
                            t2=tprev.item(),
                            class_labels=class_labels,
                            guidance_scale=guidance_scale,
                            sqrt_alphas_cumprod=sqrt_alphas_cumprod,
                            alphas_cumprod=alphas_cumprod,
                            num_classes=num_classes,
                            device=device)
        x_t = x_tprev
    return x_t