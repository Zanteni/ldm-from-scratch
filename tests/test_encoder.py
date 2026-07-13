import torch
from models.encoder import Encoder


def test_shape(encoder: Encoder, B: int = 4):
    x = torch.randn(B, 3, 32, 32, requires_grad=True)
    out = encoder(x)
    expected_shape = (B, 8, 8, 8)  # 2 * z_channels(=4), spatial 8x8
    assert out.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {out.shape}"
    print(f"[OK] shape test passed: {tuple(x.shape)} -> {tuple(out.shape)}")


def test_output_stability(encoder: Encoder, B: int = 4):
    x = torch.randn(B, 3, 32, 32, requires_grad=True)
    out = encoder(x)

    assert not torch.isnan(out).any(), "Output contains NaN values."
    assert not torch.isinf(out).any(), "Output contains infinite values."
    assert not torch.allclose(out, torch.zeros_like(out)), "Output is all zeros."
    print("[OK] output stability test passed (no NaN/Inf, not all-zero)")


def test_gradient_flow(encoder: Encoder, B: int = 4):
    x = torch.randn(B, 3, 32, 32, requires_grad=True)
    out = encoder(x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None, "No gradient reached the input."
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN values."
    assert not torch.isinf(x.grad).any(), "Input gradient contains infinite values."
    print("[OK] input gradient flow test passed")

    for name, param in encoder.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient."
        assert not torch.isnan(param.grad).any(), f"Parameter '{name}' gradient contains NaN."
        assert not torch.isinf(param.grad).any(), f"Parameter '{name}' gradient contains Inf."
    print("[OK] all parameter gradients test passed")


def run_encoder_suite(encoder: Encoder, label: str):
    print(f"===== ENCODER TEST: {label} =====")
    test_shape(encoder)
    test_output_stability(encoder)
    test_gradient_flow(encoder)
    print(f"===== {label}: ALL TESTS PASSED =====\n")


if __name__ == "__main__":
    encoder = Encoder(in_channels=3, base_channels=128, z_channels=4, num_groups=32)
    run_encoder_suite(encoder, "in=3, base=128, z=4")

    print("ALL TESTS PASSED")