import torch
from models.latent_unet_block import LatentUNetBlock


def test_shape(block, B=4, H=8, W=8, time_emb_dim=512):
    x = torch.randn(B, block.base_channels, H, W, requires_grad=True)
    t_emb = torch.randn(B, time_emb_dim, requires_grad=True)
    out = block(x, t_emb)
    expected_shape = (B, block.base_channels, H, W)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"[OK] shape test passed: {tuple(x.shape)} -> {tuple(out.shape)}")


def test_output_stability(block, B=4, H=8, W=8, time_emb_dim=512):
    x = torch.randn(B, block.base_channels, H, W, requires_grad=True)
    t_emb = torch.randn(B, time_emb_dim, requires_grad=True)
    out = block(x, t_emb)

    assert not torch.isnan(out).any(), "Output contains NaN values."
    assert not torch.isinf(out).any(), "Output contains infinite values."
    assert not torch.allclose(out, torch.zeros_like(out)), "Output is all zeros."
    print("[OK] output stability test passed (no NaN/Inf, not all-zero)")


def test_gradient_flow(block, B=4, H=8, W=8, time_emb_dim=512):
    x = torch.randn(B, block.base_channels, H, W, requires_grad=True)
    t_emb = torch.randn(B, time_emb_dim, requires_grad=True)
    out = block(x, t_emb)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None, "No gradient reached input x."
    assert not torch.isnan(x.grad).any(), "x gradient contains NaN."
    assert t_emb.grad is not None, "No gradient reached t_emb."
    assert not torch.isnan(t_emb.grad).any(), "t_emb gradient contains NaN."
    print("[OK] input gradient flow test passed (x and t_emb)")

    for name, param in block.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient."
        assert not torch.isnan(param.grad).any(), f"Parameter '{name}' gradient contains NaN."
    print("[OK] all parameter gradients test passed")


def test_composability(block, B=4, H=8, W=8, time_emb_dim=512, L=4):
    x = torch.randn(B, block.base_channels, H, W, requires_grad=True)
    t_emb = torch.randn(B, time_emb_dim, requires_grad=True)
    expected_shape = (B, block.base_channels, H, W)

    for i in range(L):
        x = block(x, t_emb)
        assert x.shape == expected_shape, f"After iteration {i}, expected {expected_shape}, got {x.shape}"
        assert not torch.isnan(x).any(), f"NaN appeared after iteration {i}."

    print(f"[OK] composability test passed: block composed with itself {L} times, shape stable throughout")


def run_latent_unet_block_suite():
    print("===== LATENT UNET BLOCK TEST =====")
    block = LatentUNetBlock(base_channels=128, time_emb_dim=512, N=6, max_mult=4)

    test_shape(block)
    test_output_stability(block)
    test_gradient_flow(block)
    test_composability(block)
    print("===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    run_latent_unet_block_suite()