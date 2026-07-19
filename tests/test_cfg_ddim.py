import torch
import torch.nn as nn

from diffusion.scheduler import get_beta_schedule, get_diffusion_constants
from diffusion.ddim import cfg_ddim_step, cfg_ddim_sample


class CountingConditionalDummyModel(nn.Module):
    """Fake conditional noise predictor that counts calls and depends on
    a real parameter, so gradient-style checks remain possible."""

    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.call_count = 0

    def forward(self, x_t, t, class_labels):
        self.call_count += 1
        return torch.randn_like(x_t) + self.dummy_param


def test_cfg_ddim_step_shape():
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    model = CountingConditionalDummyModel()
    x_t = torch.randn(4, 4, 8, 8)
    class_labels = torch.randint(0, 10, (4,))

    x_t2 = cfg_ddim_step(
        model, x_t, t1=500, t2=480,
        class_labels=class_labels, guidance_scale=3.0,
        sqrt_alphas_cumprod=constants["sqrt_alphas_cumprod"],
        alphas_cumprod=constants["alphas_cumprod"],
        num_classes=10,
    )

    assert x_t2.shape == x_t.shape, f"Expected {x_t.shape}, got {x_t2.shape}"
    print(f"[OK] cfg_ddim_step shape test passed: {tuple(x_t2.shape)}")


def test_cfg_ddim_step_stability():
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    model = CountingConditionalDummyModel()
    x_t = torch.randn(4, 4, 8, 8)
    class_labels = torch.randint(0, 10, (4,))

    x_t2 = cfg_ddim_step(
        model, x_t, t1=500, t2=480,
        class_labels=class_labels, guidance_scale=3.0,
        sqrt_alphas_cumprod=constants["sqrt_alphas_cumprod"],
        alphas_cumprod=constants["alphas_cumprod"],
        num_classes=10,
    )

    assert not torch.isnan(x_t2).any(), "Output contains NaN values."
    assert not torch.isinf(x_t2).any(), "Output contains infinite values."
    print("[OK] cfg_ddim_step stability test passed (no NaN/Inf)")


def test_cfg_ddim_sample_correct_call_count():
    """Correctness check: num_steps=50 should call the model exactly
    2*(50-1)=98 times -- one conditional + one unconditional call per
    adjacent pair in the reduced schedule."""
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    model = CountingConditionalDummyModel()
    shape = (2, 4, 8, 8)
    class_labels = torch.randint(0, 10, (2,))
    num_steps = 50

    z_0 = cfg_ddim_sample(
        model, shape, T=T, num_steps=num_steps,
        class_labels=class_labels, guidance_scale=3.0,
        sqrt_alphas_cumprod=constants["sqrt_alphas_cumprod"],
        alphas_cumprod=constants["alphas_cumprod"],
        num_classes=10,
    )

    expected_calls = 2 * (num_steps - 1)
    assert model.call_count == expected_calls, \
        f"Expected {expected_calls} model calls, got {model.call_count}"
    assert z_0.shape == shape, f"Expected {shape}, got {z_0.shape}"
    assert not torch.isnan(z_0).any(), "Output contains NaN values."
    print(f"[OK] call count test passed: {model.call_count} calls for num_steps={num_steps}")


def test_guidance_scale_zero_gives_pure_unconditional():
    """Correctness check on the blending formula: with guidance_scale=0,
    eps_pred should reduce to EXACTLY eps_uncond (the null-class prediction),
    since eps_uncond + 0*(eps_cond - eps_uncond) = eps_uncond."""
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    torch.manual_seed(0)

    class FixedOutputModel(nn.Module):
        """Returns a DIFFERENT fixed output depending on whether the label
        is the null class or not, so we can directly verify the blend."""
        def forward(self, x_t, t, class_labels):
            is_null = (class_labels == 10)
            out = torch.where(
                is_null.view(-1, 1, 1, 1),
                torch.ones_like(x_t) * 5.0,   # unconditional output
                torch.ones_like(x_t) * 100.0,  # conditional output (very different)
            )
            return out

    model = FixedOutputModel()
    x_t = torch.randn(2, 4, 8, 8)
    class_labels = torch.tensor([3, 7])

    x_t2_zero_guidance = cfg_ddim_step(
        model, x_t, t1=500, t2=480,
        class_labels=class_labels, guidance_scale=0.0,
        sqrt_alphas_cumprod=constants["sqrt_alphas_cumprod"],
        alphas_cumprod=constants["alphas_cumprod"],
        num_classes=10,
    )

    x_t2_uncond_only = cfg_ddim_step(
        model, x_t, t1=500, t2=480,
        class_labels=torch.full((2,), 10),  # force null class directly
        guidance_scale=1.0,  # irrelevant since cond==uncond when label is already null... 
        sqrt_alphas_cumprod=constants["sqrt_alphas_cumprod"],
        alphas_cumprod=constants["alphas_cumprod"],
        num_classes=10,
    )

    assert torch.allclose(x_t2_zero_guidance, x_t2_uncond_only, atol=1e-5), \
        "guidance_scale=0 did not reduce to pure unconditional prediction."
    print("[OK] guidance_scale=0 correctness test passed: reduces to pure unconditional")


def run_cfg_ddim_suite():
    print("===== CFG DDIM TEST =====")
    test_cfg_ddim_step_shape()
    test_cfg_ddim_step_stability()
    test_cfg_ddim_sample_correct_call_count()
    test_guidance_scale_zero_gives_pure_unconditional()
    print("===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    run_cfg_ddim_suite()