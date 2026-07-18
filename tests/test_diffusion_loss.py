import torch
import torch.nn as nn

from diffusion.scheduler import get_beta_schedule, get_diffusion_constants
from diffusion.diffusion_loss import compute_diffusion_loss


class DummyNoisePredictor(nn.Module):
    """Fake stand-in for the real UNet (Phase 3). Barely uses its inputs,
    but depends on a real learnable parameter so gradient flow is genuinely
    testable -- purpose is only to test compute_diffusion_loss's plumbing,
    not real learning."""

    def __init__(self):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x_t, t):
        return torch.randn_like(x_t) + self.dummy_param


def test_loss_is_finite_and_scalar():
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    model = DummyNoisePredictor()
    B = 4
    x_0 = torch.randn(B, 4, 8, 8)
    t = torch.randint(0, T, (B,))

    loss = compute_diffusion_loss(
        model, x_0, t,
        constants["sqrt_alphas_cumprod"],
        constants["sqrt_one_minus_alphas_cumprod"],
    )

    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    print(f"[OK] loss is finite and scalar test passed: loss={loss.item():.4f}")


def test_gradient_flow():
    T = 1000
    betas = get_beta_schedule("linear", T)
    constants = get_diffusion_constants(betas)

    model = DummyNoisePredictor()
    B = 4
    x_0 = torch.randn(B, 4, 8, 8)
    t = torch.randint(0, T, (B,))

    loss = compute_diffusion_loss(
        model, x_0, t,
        constants["sqrt_alphas_cumprod"],
        constants["sqrt_one_minus_alphas_cumprod"],
    )
    loss.backward()

    assert model.dummy_param.grad is not None, "No gradient reached dummy_param."
    assert not torch.isnan(model.dummy_param.grad).any(), "Gradient contains NaN."
    assert not torch.isinf(model.dummy_param.grad).any(), "Gradient contains Inf."
    print(f"[OK] gradient flow test passed: dummy_param.grad={model.dummy_param.grad.item():.4f}")


def run_diffusion_loss_suite():
    print("===== DIFFUSION LOSS TEST =====")
    test_loss_is_finite_and_scalar()
    test_gradient_flow()
    print("===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    run_diffusion_loss_suite()