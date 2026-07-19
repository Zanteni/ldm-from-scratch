import torch
from models.class_attention_block import ClassAttentionBlock


def test_shape(block, B=4, C=128, H=8, W=8, class_emb_dim=256):
    x = torch.randn(B, C, H, W, requires_grad=True)
    c = torch.randn(B, class_emb_dim, requires_grad=True)
    out = block(x, c)
    expected_shape = (B, C, H, W)
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"[OK] shape test passed: {tuple(x.shape)} -> {tuple(out.shape)}")


def test_output_stability(block, B=4, C=128, H=8, W=8, class_emb_dim=256):
    x = torch.randn(B, C, H, W, requires_grad=True)
    c = torch.randn(B, class_emb_dim, requires_grad=True)
    out = block(x, c)

    assert not torch.isnan(out).any(), "Output contains NaN values."
    assert not torch.isinf(out).any(), "Output contains infinite values."
    assert not torch.allclose(out, torch.zeros_like(out)), "Output is all zeros."
    print("[OK] output stability test passed (no NaN/Inf, not all-zero)")


def test_gradient_flow(block, B=4, C=128, H=8, W=8, class_emb_dim=256):
    x = torch.randn(B, C, H, W, requires_grad=True)
    c = torch.randn(B, class_emb_dim, requires_grad=True)
    out = block(x, c)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None, "No gradient reached input x."
    assert not torch.isnan(x.grad).any(), "x gradient contains NaN."
    print("[OK] input x gradient flow test passed")

    assert c.grad is not None, "No gradient reached input c."
    assert not torch.isnan(c.grad).any(), "c gradient contains NaN."
    print("[OK] input c gradient flow test passed")

    for name, param in block.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient."
        assert not torch.isnan(param.grad).any(), f"Parameter '{name}' gradient contains NaN."
    print("[OK] all parameter gradients test passed")


def test_different_class_gives_different_output(block, B=4, C=128, H=8, W=8, class_emb_dim=256):
    """Correctness check: class conditioning must actually matter -- same
    x, different c, should produce different output. If identical, the
    class token isn't influencing anything (concatenation/slicing bug)."""
    x = torch.randn(B, C, H, W)
    c1 = torch.randn(B, class_emb_dim)
    c2 = torch.randn(B, class_emb_dim)

    out1 = block(x, c1)
    out2 = block(x, c2)

    assert not torch.allclose(out1, out2), \
        "Different class embeddings produced identical output -- class conditioning may be broken."
    print("[OK] different class -> different output test passed (class conditioning is load-bearing)")


def run_class_attention_block_suite():
    print("===== CLASS ATTENTION BLOCK TEST =====")
    block = ClassAttentionBlock(channels=128, class_emb_dim=256, num_groups=32, num_heads=4)

    test_shape(block)
    test_output_stability(block)
    test_gradient_flow(block)
    test_different_class_gives_different_output(block)
    print("===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    run_class_attention_block_suite()