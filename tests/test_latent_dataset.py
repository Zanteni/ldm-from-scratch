import torch
from torch.utils.data import DataLoader, TensorDataset

from models.vae import VAE
from dataset import build_latent_cache, LatentDataset


def test_build_latent_cache_shapes():
    vae = VAE()
    N = 10
    fake_images = torch.randn(N, 3, 32, 32)
    fake_dataset = TensorDataset(fake_images)
    dataloader = DataLoader(fake_dataset, batch_size=4, shuffle=False)

    all_mu, all_logvar = build_latent_cache(vae, dataloader, device="cpu")

    assert all_mu.shape == (N, 4, 8, 8), f"Expected mu shape ({N},4,8,8), got {all_mu.shape}"
    assert all_logvar.shape == (N, 4, 8, 8), f"Expected logvar shape ({N},4,8,8), got {all_logvar.shape}"
    print(f"[OK] build_latent_cache shape test passed: mu={tuple(all_mu.shape)}, logvar={tuple(all_logvar.shape)}")

    return all_mu, all_logvar


def test_latent_dataset_len(all_mu, all_logvar):
    ds = LatentDataset(all_mu, all_logvar)
    assert len(ds) == all_mu.shape[0], f"Expected len {all_mu.shape[0]}, got {len(ds)}"
    print(f"[OK] LatentDataset __len__ test passed: {len(ds)}")


def test_latent_dataset_getitem_shape_and_stability(all_mu, all_logvar):
    ds = LatentDataset(all_mu, all_logvar)
    sample = ds[0]

    assert isinstance(sample, tuple) and len(sample) == 1, f"Expected (z,), got {sample}"
    z = sample[0]

    assert z.shape == (4, 8, 8), f"Expected z shape (4,8,8), got {z.shape}"
    assert not torch.isnan(z).any(), "z contains NaN values."
    assert not torch.isinf(z).any(), "z contains infinite values."
    print(f"[OK] LatentDataset __getitem__ shape/stability test passed: z shape={tuple(z.shape)}")


def test_latent_dataset_fresh_sampling(all_mu, all_logvar):
    ds = LatentDataset(all_mu, all_logvar)

    z1, = ds[0]
    z2, = ds[0]

    assert not torch.allclose(z1, z2), \
        "Two calls to __getitem__(0) returned identical z -- reparameterization may not be resampling."
    print("[OK] fresh sampling test passed: repeated __getitem__ calls produce different z")


def run_latent_dataset_suite():
    print("===== LATENT DATASET TEST =====")
    all_mu, all_logvar = test_build_latent_cache_shapes()
    test_latent_dataset_len(all_mu, all_logvar)
    test_latent_dataset_getitem_shape_and_stability(all_mu, all_logvar)
    test_latent_dataset_fresh_sampling(all_mu, all_logvar)
    print("===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    run_latent_dataset_suite()