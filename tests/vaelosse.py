import torch
from diffusion.losses import VAELoss
from models.vae import VAE


def test_loss_is_finite(vae: VAE, loss_fn: VAELoss, B: int = 4):
    x = torch.randn(B, 3, 32, 32)
    out = vae(x)
    total, parts = loss_fn(out["recon"], x, out["mu"], out["logvar"])

    assert torch.isfinite(total), f"Loss is not finite: {total}"
    print(f"[OK] loss is finite test passed: {parts}")


def test_loss_decreases_with_optimization(steps: int = 10, batch_size: int = 8, lr: float = 1e-3):
    torch.manual_seed(0)
    target = torch.randn(batch_size, 3, 32, 32)
    vae = VAE()
    loss_fn = VAELoss()
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    losses = []
    for step in range(steps):
        optimizer.zero_grad()

        out = vae(target)
        total_loss, parts = loss_fn(out["recon"], target, out["mu"], out["logvar"])

        total_loss.backward()
        optimizer.step()

        losses.append(total_loss.item())
        print(f"  step {step}: loss={total_loss.item():.4f}")

    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]} -> {losses[-1]}"
    print(f"[OK] loss decreased test passed: {losses[0]:.4f} -> {losses[-1]:.4f}")


def run_loss_suite():
    print("===== VAELOSS TEST =====")
    vae = VAE()
    loss_fn = VAELoss()

    test_loss_is_finite(vae, loss_fn)
    test_loss_decreases_with_optimization()
    print("===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    run_loss_suite()