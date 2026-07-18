import torch
from models.resblock import TimeResBlock


def test_shape(block: TimeResBlock, B: int = 4, H: int = 8, W: int = 8, time_emb_dim: int = 512):
    x = torch.randn(B, block.in_channels, H, W, requires_grad=True)
    t_emb = torch.randn(B, time_emb_dim, requires_grad=True)
    out = block(x, t_emb)
    expected_shape = (B, block.out_channels, H, W)
    assert out.shape == expected_shape, f"Expected shape {expected_shape}, got {out.shape}"
    print(f"[OK] shape test passed: {tuple(x.shape)} -> {tuple(out.shape)}")


def test_output_stability(block: TimeResBlock, B: int = 4, H: int = 8, W: int = 8, time_emb_dim: int = 512):
    x = torch.randn(B, block.in_channels, H, W, requires_grad=True)
    t_emb = torch.randn(B, time_emb_dim, requires_grad=True)
    out = block(x, t_emb)

    assert not torch.isnan(out).any(), "Output contains NaN values."
    assert not torch.isinf(out).any(), "Output contains infinite values."
    assert not torch.allclose(out, torch.zeros_like(out)), "Output is all zeros."
    print("[OK] output stability test passed (no NaN/Inf, not all-zero)")


def test_gradient_flow(block: TimeResBlock, B: int = 4, H: int = 8, W: int = 8, time_emb_dim: int = 512):
    x = torch.randn(B, block.in_channels, H, W, requires_grad=True)
    t_emb = torch.randn(B, time_emb_dim, requires_grad=True)
    out = block(x, t_emb)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None, "No gradient reached input x."
    assert not torch.isnan(x.grad).any(), "x gradient contains NaN values."
    assert not torch.isinf(x.grad).any(), "x gradient contains infinite values."
    print("[OK] input x gradient flow test passed")

    assert t_emb.grad is not None, "No gradient reached t_emb."
    assert not torch.isnan(t_emb.grad).any(), "t_emb gradient contains NaN values."
    assert not torch.isinf(t_emb.grad).any(), "t_emb gradient contains infinite values."
    print("[OK] input t_emb gradient flow test passed")

    for name, param in block.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient."
        assert not torch.isnan(param.grad).any(), f"Parameter '{name}' gradient contains NaN."
        assert not torch.isinf(param.grad).any(), f"Parameter '{name}' gradient contains Inf."
    print("[OK] all parameter gradients test passed (including time_proj)")


def test_skip_path_type(block: TimeResBlock):
    if block.in_channels == block.out_channels:
        assert isinstance(block.skip, torch.nn.Identity), \
            "Expected Identity skip when in_channels == out_channels"
        print("[OK] skip path is Identity (same-channel case)")
    else:
        assert not isinstance(block.skip, torch.nn.Identity), \
            "Expected a projection Conv skip when in_channels != out_channels"
        print("[OK] skip path is a projection conv (different-channel case)")


def run_time_resblock_suite(block: TimeResBlock, label: str):
    print(f"===== TIME RESBLOCK TEST: {label} =====")
    test_shape(block)
    test_output_stability(block)
    test_gradient_flow(block)
    test_skip_path_type(block)
    print(f"===== {label}: ALL TESTS PASSED =====\n")


if __name__ == "__main__":
    block_same = TimeResBlock(in_channels=128, out_channels=128, time_emb_dim=512)
    run_time_resblock_suite(block_same, "same-channel (128->128)")

    block_diff = TimeResBlock(in_channels=128, out_channels=256, time_emb_dim=512)
    run_time_resblock_suite(block_diff, "different-channel (128->256)")

    print("ALL TESTS PASSED")