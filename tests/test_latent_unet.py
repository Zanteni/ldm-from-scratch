import torch
from models.latent_unet import LatentUNet


def test_shape(model, B=4, z_channels=4, H=8, W=8):
    z_t = torch.randn(B, z_channels, H, W, requires_grad=True)
    t = torch.randint(0, 1000, (B,))
    out = model(z_t, t)

    expected_shape = (B, z_channels, H, W)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"[OK] shape test passed: {tuple(z_t.shape)} -> {tuple(out.shape)}")


def test_output_stability(model, B=4, z_channels=4, H=8, W=8):
    z_t = torch.randn(B, z_channels, H, W, requires_grad=True)
    t = torch.randint(0, 1000, (B,))
    out = model(z_t, t)

    assert not torch.isnan(out).any(), "Output contains NaN values."
    assert not torch.isinf(out).any(), "Output contains infinite values."
    assert not torch.allclose(out, torch.zeros_like(out)), "Output is all zeros."
    print("[OK] output stability test passed (no NaN/Inf, not all-zero)")


def test_gradient_flow(model, B=4, z_channels=4, H=8, W=8):
    z_t = torch.randn(B, z_channels, H, W, requires_grad=True)
    t = torch.randint(0, 1000, (B,))
    out = model(z_t, t)
    loss = out.mean()
    loss.backward()

    assert z_t.grad is not None, "No gradient reached input z_t."
    assert not torch.isnan(z_t.grad).any(), "z_t gradient contains NaN values."
    assert not torch.isinf(z_t.grad).any(), "z_t gradient contains infinite values."
    print("[OK] input z_t gradient flow test passed")

    missing = []
    for name, param in model.named_parameters():
        if param.grad is None:
            missing.append(name)
        elif torch.isnan(param.grad).any():
            raise AssertionError(f"Parameter '{name}' gradient contains NaN.")
        elif torch.isinf(param.grad).any():
            raise AssertionError(f"Parameter '{name}' gradient contains Inf.")

    assert not missing, f"These parameters received no gradient: {missing}"
    print(f"[OK] all parameter gradients test passed ({sum(1 for _ in model.parameters())} parameter tensors checked)")


def test_different_t_gives_different_output(model, B=2, z_channels=4, H=8, W=8):
    """Sanity check: the model should actually use t, not ignore it --
    same z_t with two very different t values should give different eps_pred."""
    torch.manual_seed(0)
    z_t = torch.randn(1, z_channels, H, W).repeat(2, 1, 1, 1)
    t = torch.tensor([0, 999])

    out = model(z_t, t)
    assert not torch.allclose(out[0], out[1]), \
        "Model produced identical output for very different t -- may be ignoring the timestep."
    print("[OK] different t -> different output test passed (model actually uses t)")


def run_latent_unet_suite():
    print("===== LATENT UNET TEST =====")
    model = LatentUNet(z_channels=4, base_channels=128, time_emb_dim=512, N=6, L=4, max_mult=4)

    test_shape(model)
    test_output_stability(model)
    test_gradient_flow(model)
    test_different_t_gives_different_output(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    run_latent_unet_suite()