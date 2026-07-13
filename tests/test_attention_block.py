import torch
from models.attention_block import AttentionBlock

def test_shape(attn_block: AttentionBlock, B: int = 4, C: int = 256, H: int = 8, W: int = 8):
    x = torch.randn(B, C, H, W, requires_grad=True)
    out = attn_block(x)
    assert out.shape == (B, C, H, W), \
        f"Expected shape {(B, C, H, W)}, got {out.shape}"
    print(f"[OK] shape test passed: {tuple(x.shape)} -> {tuple(out.shape)}")


def test_output_stability(attn_block: AttentionBlock, B: int = 4, C: int = 256, H: int = 8, W: int = 8):
    x = torch.randn(B, C, H, W, requires_grad=True)
    out = attn_block(x)

    assert not torch.isnan(out).any(), "Output contains NaN values."
    assert not torch.isinf(out).any(), "Output contains infinite values."
    assert not torch.allclose(out, torch.zeros_like(out)), "Output is all zeros."
    print("[OK] output stability test passed (no NaN/Inf, not all-zero)")


def test_gradient_flow(attn_block: AttentionBlock, B: int = 4, C: int = 256, H: int = 8, W: int = 8):
    x = torch.randn(B, C, H, W, requires_grad=True)
    out = attn_block(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None, "No gradient reached the input."
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN values."
    assert not torch.isinf(x.grad).any(), "Input gradient contains infinite values."
    print("[OK] input gradient flow test passed")

    for name, param in attn_block.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient."
        assert not torch.isnan(param.grad).any(), f"Parameter '{name}' gradient contains NaN."
        assert not torch.isinf(param.grad).any(), f"Parameter '{name}' gradient contains Inf."
    print("[OK] all parameter gradients test passed")


def test_residual_not_identity(attn_block: AttentionBlock, B: int = 4, C: int = 256, H: int = 8, W: int = 8):
    """Sanity check: output should differ from raw input (attention actually
    contributes something), while still being residual (not wildly different)."""
    x = torch.randn(B, C, H, W)
    out = attn_block(x)
    assert not torch.allclose(out, x), \
        "Output is identical to input -- attention branch may be contributing nothing."
    print("[OK] residual contribution test passed (output != raw input)")


def run_attention_block_suite(attn_block: AttentionBlock, label: str):
    print(f"===== ATTENTION BLOCK TEST: {label} =====")
    test_shape(attn_block)
    test_output_stability(attn_block)
    test_gradient_flow(attn_block)
    test_residual_not_identity(attn_block)
    print(f"===== {label}: ALL TESTS PASSED =====\n")


if __name__ == "__main__":
    block = AttentionBlock(channels=256, num_groups=32, num_heads=4)
    run_attention_block_suite(block, "channels=256, heads=4")

    print("ALL TESTS PASSED")