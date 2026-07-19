import torch
import torch.nn as nn

from diffusion.scheduler import get_beta_schedule, get_diffusion_constants
from diffusion.ddim import ddim_step, ddim_sample


class CountingDummyModel(nn.Module):
    """Fake noise predictor that counts how many times it's been called,
    and depends on a real parameter so gradient-style checks would still
    be possible (not used here, but keeps the pattern consistent)."""

    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.call_count = 0

    def forward(self, x_t, t):
        self.call_count += 1
        return torch.randn_like(x_t) + self.dummy_param


def test_ddim_step_shape():
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    model = CountingDummyModel()
    x_t = torch.randn(4, 4, 8, 8)

    x_t2 = ddim_step(
        model, x_t, t1=500, t2=480,
        sqrt_alphas_cumprod=constants["sqrt_alphas_cumprod"],
        alphas_cumprod=constants["alphas_cumprod"],
    )

    assert x_t2.shape == x_t.shape, f"Expected {x_t.shape}, got {x_t2.shape}"
    print(f"[OK] ddim_step shape test passed: {tuple(x_t2.shape)}")


def test_ddim_step_stability():
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    model = CountingDummyModel()
    x_t = torch.randn(4, 4, 8, 8)

    x_t2 = ddim_step(
        model, x_t, t1=500, t2=480,
        sqrt_alphas_cumprod=constants["sqrt_alphas_cumprod"],
        alphas_cumprod=constants["alphas_cumprod"],
    )

    assert not torch.isnan(x_t2).any(), "Output contains NaN values."
    assert not torch.isinf(x_t2).any(), "Output contains infinite values."
    print("[OK] ddim_step stability test passed (no NaN/Inf)")


def test_ddim_sample_shape():
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    model = CountingDummyModel()
    shape = (2, 4, 8, 8)

    z_0 = ddim_sample(
        model, shape, T=T, num_steps=50,
        sqrt_alphas_cumprod=constants["sqrt_alphas_cumprod"],
        alphas_cumprod=constants["alphas_cumprod"],
    )

    assert z_0.shape == shape, f"Expected {shape}, got {z_0.shape}"
    assert not torch.isnan(z_0).any(), "Output contains NaN values."
    assert not torch.isinf(z_0).any(), "Output contains infinite values."
    print(f"[OK] ddim_sample shape/stability test passed: {tuple(z_0.shape)}")


def test_ddim_sample_correct_step_count():
    """Correctness check: num_steps=50 should call the model exactly 49
    times (one call per adjacent pair in the reduced schedule, not 50)."""
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    model = CountingDummyModel()
    shape = (2, 4, 8, 8)
    num_steps = 50

    ddim_sample(
        model, shape, T=T, num_steps=num_steps,
        sqrt_alphas_cumprod=constants["sqrt_alphas_cumprod"],
        alphas_cumprod=constants["alphas_cumprod"],
    )

    expected_calls = num_steps - 1
    assert model.call_count == expected_calls, \
        f"Expected {expected_calls} model calls, got {model.call_count}"
    print(f"[OK] step count test passed: {model.call_count} model calls for num_steps={num_steps}")


def run_ddim_suite():
    print("===== DDIM TEST =====")
    test_ddim_step_shape()
    test_ddim_step_stability()
    test_ddim_sample_shape()
    test_ddim_sample_correct_step_count()
    print("===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    run_ddim_suite()