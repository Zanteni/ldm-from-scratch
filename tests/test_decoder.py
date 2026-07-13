import torch
from models.decoder import Decoder


def test_shape(decoder: Decoder, B: int = 4):
    z = torch.randn(B, 4, 8, 8, requires_grad=True)
    out = decoder(z)
    expected_shape = (B, 3, 32, 32)
    assert out.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {out.shape}"
    print(f"[OK] shape test passed: {tuple(z.shape)} -> {tuple(out.shape)}")


def test_output_stability(decoder: Decoder, B: int = 4):
    z = torch.randn(B, 4, 8, 8, requires_grad=True)
    out = decoder(z)

    assert not torch.isnan(out).any(), "Output contains NaN values."
    assert not torch.isinf(out).any(), "Output contains infinite values."
    assert not torch.allclose(out, torch.zeros_like(out)), "Output is all zeros."
    print("[OK] output stability test passed (no NaN/Inf, not all-zero)")


def test_gradient_flow(decoder: Decoder, B: int = 4):
    z = torch.randn(B, 4, 8, 8, requires_grad=True)
    out = decoder(z)
    loss = out.mean()
    loss.backward()

    assert z.grad is not None, "No gradient reached the input."
    assert not torch.isnan(z.grad).any(), "Input gradient contains NaN values."
    assert not torch.isinf(z.grad).any(), "Input gradient contains infinite values."
    print("[OK] input gradient flow test passed")

    for name, param in decoder.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient."
        assert not torch.isnan(param.grad).any(), f"Parameter '{name}' gradient contains NaN."
        assert not torch.isinf(param.grad).any(), f"Parameter '{name}' gradient contains Inf."
    print("[OK] all parameter gradients test passed")


def run_decoder_suite(decoder: Decoder, label: str):
    print(f"===== DECODER TEST: {label} =====")
    test_shape(decoder)
    test_output_stability(decoder)
    test_gradient_flow(decoder)
    print(f"===== {label}: ALL TESTS PASSED =====\n")


if __name__ == "__main__":
    decoder = Decoder(out_channels=3, base_channels=128, z_channels=4, num_groups=32)
    run_decoder_suite(decoder, "out=3, base=128, z=4")

    print("ALL TESTS PASSED")