import torch
from models.resblock import ResBlock, Downsample, Upsample


def test_shape(res_block: ResBlock, B: int = 32, H: int = 16, W: int = 16):
    in_chan = res_block.in_channels
    out_chan = res_block.out_channels
    h = torch.randn(B, in_chan, H, W, requires_grad=True)
    out = res_block(h)
    assert out.shape == (B, out_chan, H, W), \
        f"Expected shape {(B, out_chan, H, W)}, got {out.shape}"
    print(f"[OK] shape test passed: {tuple(h.shape)} -> {tuple(out.shape)}")


def test_output_stability(res_block: ResBlock, B: int = 32, H: int = 16, W: int = 16):
    in_chan = res_block.in_channels
    h = torch.randn(B, in_chan, H, W, requires_grad=True)
    out = res_block(h)

    assert not torch.isnan(out).any(), "Output contains NaN values."
    assert not torch.isinf(out).any(), "Output contains infinite values."
    assert not torch.allclose(out, torch.zeros_like(out)), "Output is all zeros."
    print("[OK] output stability test passed (no NaN/Inf, not all-zero)")


def test_gradient_flow(res_block: ResBlock, B: int = 32, H: int = 16, W: int = 16):
    in_chan = res_block.in_channels
    h = torch.randn(B, in_chan, H, W, requires_grad=True)
    out = res_block(h)
    loss = out.mean()
    loss.backward()

    assert h.grad is not None, "No gradient reached the input."
    assert not torch.isnan(h.grad).any(), "Input gradient contains NaN values."
    assert not torch.isinf(h.grad).any(), "Input gradient contains infinite values."
    print("[OK] input gradient flow test passed")

    for name, param in res_block.named_parameters():
        assert param.grad is not None, f"Parameter '{name}' received no gradient."
        assert not torch.isnan(param.grad).any(), f"Parameter '{name}' gradient contains NaN."
        assert not torch.isinf(param.grad).any(), f"Parameter '{name}' gradient contains Inf."
    print("[OK] all parameter gradients test passed")


def test_skip_path_type(res_block: ResBlock):
    if res_block.in_channels == res_block.out_channels:
        assert isinstance(res_block.skip, torch.nn.Identity), \
            "Expected Identity skip when in_channels == out_channels"
        print("[OK] skip path is Identity (same-channel case)")
    else:
        assert not isinstance(res_block.skip, torch.nn.Identity), \
            "Expected a projection Conv skip when in_channels != out_channels"
        print("[OK] skip path is a projection conv (different-channel case)")


def run_resblock_suite(res_block: ResBlock, label: str):
    print(f"===== RESBLOCK TEST: {label} =====")
    test_shape(res_block)
    test_output_stability(res_block)
    test_gradient_flow(res_block)
    test_skip_path_type(res_block)
    print(f"===== {label}: ALL TESTS PASSED =====\n")


def test_downsample():
    print("===== DOWNSAMPLE TEST =====")
    x = torch.randn(2, 128, 32, 32, requires_grad=True)
    down = Downsample(channels=128)
    out = down(x)
    assert out.shape == (2, 128, 16, 16), f"Expected (2,128,16,16), got {out.shape}"
    print(f"[OK] downsample shape test passed: {tuple(x.shape)} -> {tuple(out.shape)}")

    loss = out.mean()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any(), "Downsample gradient broken."
    print("[OK] downsample gradient flow test passed")
    print("===== DOWNSAMPLE: ALL TESTS PASSED =====\n")


def test_upsample():
    print("===== UPSAMPLE TEST =====")
    x = torch.randn(2, 128, 16, 16, requires_grad=True)
    up = Upsample(channels=128)
    out = up(x)
    assert out.shape == (2, 128, 32, 32), f"Expected (2,128,32,32), got {out.shape}"
    print(f"[OK] upsample shape test passed: {tuple(x.shape)} -> {tuple(out.shape)}")

    loss = out.mean()
    loss.backward()
    assert x.grad is not None and not torch.isnan(x.grad).any(), "Upsample gradient broken."
    print("[OK] upsample gradient flow test passed")
    print("===== UPSAMPLE: ALL TESTS PASSED =====\n")


if __name__ == "__main__":
    res_same = ResBlock(in_channels=128, out_channels=128)
    run_resblock_suite(res_same, "same-channel (128->128)")

    res_diff = ResBlock(in_channels=128, out_channels=256)
    run_resblock_suite(res_diff, "different-channel (128->256)")

    test_downsample()
    test_upsample()

    print("ALL TESTS PASSED")