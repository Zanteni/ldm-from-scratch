import torch
from models.latent_unet import ConditionalLatentUNet


def test_shape(model, B=4, z_channels=4, H=8, W=8, num_classes=10):
    z_t = torch.randn(B, z_channels, H, W, requires_grad=True)
    t = torch.randint(0, 1000, (B,))
    class_labels = torch.randint(0, num_classes, (B,))

    out = model(z_t, t, class_labels)
    expected_shape = (B, z_channels, H, W)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"[OK] shape test passed: {tuple(z_t.shape)} -> {tuple(out.shape)}")


def test_output_stability(model, B=4, z_channels=4, H=8, W=8, num_classes=10):
    z_t = torch.randn(B, z_channels, H, W, requires_grad=True)
    t = torch.randint(0, 1000, (B,))
    class_labels = torch.randint(0, num_classes, (B,))

    out = model(z_t, t, class_labels)
    assert not torch.isnan(out).any(), "Output contains NaN values."
    assert not torch.isinf(out).any(), "Output contains infinite values."
    assert not torch.allclose(out, torch.zeros_like(out)), "Output is all zeros."
    print("[OK] output stability test passed (no NaN/Inf, not all-zero)")


def test_gradient_flow(model, B=4, z_channels=4, H=8, W=8, num_classes=10):
    z_t = torch.randn(B, z_channels, H, W, requires_grad=True)
    t = torch.randint(0, 1000, (B,))
    class_labels = torch.randint(0, num_classes, (B,))

    out = model(z_t, t, class_labels)
    loss = out.mean()
    loss.backward()

    assert z_t.grad is not None, "No gradient reached input z_t."
    assert not torch.isnan(z_t.grad).any(), "z_t gradient contains NaN values."
    print("[OK] input z_t gradient flow test passed")

    missing = []
    for name, param in model.named_parameters():
        if param.grad is None:
            missing.append(name)
        elif torch.isnan(param.grad).any():
            raise AssertionError(f"Parameter '{name}' gradient contains NaN.")

    assert not missing, f"These parameters received no gradient: {missing}"
    print(f"[OK] all parameter gradients test passed ({sum(1 for _ in model.parameters())} parameter tensors checked)")


def test_different_class_gives_different_output(model, B=2, z_channels=4, H=8, W=8):
    """Same z_t and t, but two different class labels, should produce
    different output -- confirms class conditioning is load-bearing at
    the full-model level, not just within ClassAttentionBlock alone."""
    torch.manual_seed(0)
    z_t = torch.randn(1, z_channels, H, W).repeat(2, 1, 1, 1)
    t = torch.full((2,), 500, dtype=torch.long)
    class_labels = torch.tensor([0, 5])  # two different real classes

    out = model(z_t, t, class_labels)
    assert not torch.allclose(out[0], out[1]), \
        "Different class labels produced identical output -- class conditioning may be broken."
    print("[OK] different class -> different output test passed")


def test_null_class_works(model, B=4, z_channels=4, H=8, W=8, num_classes=10):
    """Passing the reserved null-class index (num_classes) should work
    without error and produce valid output -- needed for CFG later."""
    z_t = torch.randn(B, z_channels, H, W)
    t = torch.randint(0, 1000, (B,))
    null_labels = torch.full((B,), num_classes, dtype=torch.long)  # index 10 for num_classes=10

    out = model(z_t, t, null_labels)
    assert out.shape == (B, z_channels, H, W), f"Expected {(B,z_channels,H,W)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Null-class output contains NaN values."
    print("[OK] null class test passed: model runs correctly with the reserved null-class index")


def run_conditional_latent_unet_suite():
    print("===== CONDITIONAL LATENT UNET TEST =====")
    model = ConditionalLatentUNet(
        z_channels=4, base_channels=128, time_emb_dim=512,
        num_classes=10, class_emb_dim=256,
        N=6, L=4, max_mult=4,
    )

    test_shape(model)
    test_output_stability(model)
    test_gradient_flow(model)
    test_different_class_gives_different_output(model)
    test_null_class_works(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    run_conditional_latent_unet_suite()