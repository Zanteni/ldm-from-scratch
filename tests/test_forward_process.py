import torch
from diffusion.scheduler import get_beta_schedule, get_diffusion_constants
from diffusion.forward_process import forward_diffusion_sample


def test_output_shapes():
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    B, C, H, W = 4, 4, 8, 8
    x_0 = torch.randn(B, C, H, W)
    t = torch.randint(0, T, (B,))

    x_t, eps = forward_diffusion_sample(
        x_0, t, constants["sqrt_alphas_cumprod"], constants["sqrt_one_minus_alphas_cumprod"]
    )

    assert x_t.shape == (B, C, H, W), f"Expected x_t shape {(B,C,H,W)}, got {x_t.shape}"
    assert eps.shape == (B, C, H, W), f"Expected eps shape {(B,C,H,W)}, got {eps.shape}"
    print(f"[OK] output shapes test passed: x_t={tuple(x_t.shape)}, eps={tuple(eps.shape)}")


def test_output_stability():
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    x_0 = torch.randn(4, 4, 8, 8)
    t = torch.randint(0, T, (4,))

    x_t, eps = forward_diffusion_sample(
        x_0, t, constants["sqrt_alphas_cumprod"], constants["sqrt_one_minus_alphas_cumprod"]
    )

    assert not torch.isnan(x_t).any(), "x_t contains NaN values."
    assert not torch.isinf(x_t).any(), "x_t contains infinite values."
    print("[OK] output stability test passed (no NaN/Inf)")


def test_t_zero_stays_close_to_x0():
    """At t=0, alpha_bar_0 should be very close to 1, so x_t should be
    almost identical to x_0 (minimal noise added)."""
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    x_0 = torch.randn(4, 4, 8, 8)
    t = torch.zeros(4, dtype=torch.long)

    x_t, eps = forward_diffusion_sample(
        x_0, t, constants["sqrt_alphas_cumprod"], constants["sqrt_one_minus_alphas_cumprod"]
    )

    diff = (x_t - x_0).abs().mean().item()
    assert diff < 0.5, f"Expected x_t close to x_0 at t=0, got mean abs diff {diff:.4f}"
    print(f"[OK] t=0 stays close to x_0 test passed: mean abs diff={diff:.4f}")


def test_t_max_is_mostly_noise():
    """At the largest t, alpha_bar_t should be close to 0, so x_t should be
    dominated by eps rather than x_0."""
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    x_0 = torch.ones(4, 4, 8, 8) * 10.0  # deliberately large signal
    t = torch.full((4,), T - 1, dtype=torch.long)

    x_t, eps = forward_diffusion_sample(
        x_0, t, constants["sqrt_alphas_cumprod"], constants["sqrt_one_minus_alphas_cumprod"]
    )

    # x_t should look much more like eps than like the huge x_0 signal
    diff_from_x0 = (x_t - x_0).abs().mean().item()
    diff_from_eps = (x_t - eps).abs().mean().item()
    assert diff_from_eps < diff_from_x0, \
        f"Expected x_t closer to eps than x_0 at t=T-1, got diff_from_eps={diff_from_eps:.4f}, diff_from_x0={diff_from_x0:.4f}"
    print(f"[OK] t=T-1 mostly noise test passed: diff_from_eps={diff_from_eps:.4f} < diff_from_x0={diff_from_x0:.4f}")


def test_different_t_per_batch_sample():
    """Confirm each sample in the batch can be noised to a genuinely
    different timestep, not one shared t for the whole batch."""
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    x_0 = torch.randn(4, 4, 8, 8)
    t = torch.tensor([0, 250, 500, 999])

    x_t, eps = forward_diffusion_sample(
        x_0, t, constants["sqrt_alphas_cumprod"], constants["sqrt_one_minus_alphas_cumprod"]
    )

    # sample 0 (t=0) should stay close to x_0; sample 3 (t=999) should not
    diff_t0 = (x_t[0] - x_0[0]).abs().mean().item()
    diff_t999 = (x_t[3] - x_0[3]).abs().mean().item()
    assert diff_t0 < diff_t999, \
        f"Expected less noise at t=0 than t=999, got diff_t0={diff_t0:.4f}, diff_t999={diff_t999:.4f}"
    print(f"[OK] different t per batch sample test passed: diff_t0={diff_t0:.4f} < diff_t999={diff_t999:.4f}")


def run_forward_process_suite():
    print("===== FORWARD DIFFUSION PROCESS TEST =====")
    test_output_shapes()
    test_output_stability()
    test_t_zero_stays_close_to_x0()
    test_t_max_is_mostly_noise()
    test_different_t_per_batch_sample()
    print("===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    run_forward_process_suite()