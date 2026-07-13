import torch

from models.vae import VAE

def test_forward_shapes(vae: VAE, B: int = 4):
    x = torch.randn(B, 3, 32, 32, requires_grad=True)
    out = vae(x)
    assert out["recon"].shape == (B, 3, 32, 32), \
        f"recon shape wrong: {out['recon'].shape}"
    assert out["mu"].shape == (B, 4, 8, 8), \
        f"mu shape wrong: {out['mu'].shape}"
    assert out["logvar"].shape == (B, 4, 8, 8), \
        f"logvar shape wrong: {out['logvar'].shape}"
    print(f"[OK] forward shapes test passed: recon={tuple(out['recon'].shape)}, "
          f"mu={tuple(out['mu'].shape)}, logvar={tuple(out['logvar'].shape)}")


def test_output_stability(vae: VAE, B: int = 4):
    x = torch.randn(B, 3, 32, 32, requires_grad=True)
    out = vae(x)

    for name in ["recon", "mu", "logvar"]:
        t = out[name]
        assert not torch.isnan(t).any(), f"{name} contains NaN values."
        assert not torch.isinf(t).any(), f"{name} contains infinite values."
        assert not torch.allclose(t, torch.zeros_like(t)), f"{name} is all zeros."
    print("[OK] output stability test passed (no NaN/Inf, not all-zero) for recon/mu/logvar")


def test_gradient_flow(vae: VAE, B: int = 4):
    x = torch.randn(B, 3, 32, 32, requires_grad=True)
    out = vae(x)
    loss = out["recon"].mean() + out["kl"]
    loss.backward()

    assert x.grad is not None, "No gradient reached the input."
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN values."
    assert not torch.isinf(x.grad).any(), "Input gradient contains infinite values."
    print("[OK] input gradient flow test passed")

    for name, param in vae.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient."
        assert not torch.isnan(param.grad).any(), f"Parameter '{name}' gradient contains NaN."
        assert not torch.isinf(param.grad).any(), f"Parameter '{name}' gradient contains Inf."
    print("[OK] all parameter gradients test passed")


def test_kl_zero_at_standard_normal():
    mu = torch.zeros(2, 4, 8, 8)
    logvar = torch.zeros(2, 4, 8, 8)
    kl = VAE.kl_divergence(mu, logvar)
    assert kl.item() < 1e-5, f"Expected KL near 0, got {kl.item()}"
    print(f"[OK] KL zero-at-standard-normal test passed: KL={kl.item():.8f}")


def test_kl_nonneg_and_finite():
    mu = torch.randn(4, 4, 8, 8) * 5  # arbitrary, non-trivial mu/logvar
    logvar = torch.randn(4, 4, 8, 8)
    kl = VAE.kl_divergence(mu, logvar)
    assert torch.isfinite(kl), f"KL is not finite: {kl}"
    assert kl.item() >= 0, f"KL should be >= 0, got {kl.item()}"
    print(f"[OK] KL non-negative and finite test passed: KL={kl.item():.4f}")


def run_vae_suite(vae: VAE, label: str):
    print(f"===== VAE TEST: {label} =====")
    test_forward_shapes(vae)
    test_output_stability(vae)
    test_gradient_flow(vae)
    test_kl_zero_at_standard_normal()
    test_kl_nonneg_and_finite()
    print(f"===== {label}: ALL TESTS PASSED =====\n")


if __name__ == "__main__":
    vae = VAE(in_channels=3, base_channels=128, z_channels=4, num_groups=32)
    run_vae_suite(vae, "in=3, base=128, z=4")

    print("ALL TESTS PASSED")