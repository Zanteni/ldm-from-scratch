import torch
from torch.utils.data import DataLoader, TensorDataset

from models.vae import VAE
from dataset import build_conditional_latent_cache, ConditionalLatentDataset


def test_build_conditional_latent_cache_shapes_and_labels():
    vae = VAE()
    N = 10
    fake_images = torch.randn(N, 3, 32, 32)
    fake_labels = torch.arange(N)  # 0,1,2,...,9 -- distinct, easy to verify order
    fake_dataset = TensorDataset(fake_images, fake_labels)
    dataloader = DataLoader(fake_dataset, batch_size=4, shuffle=False)

    all_mu, all_logvar, all_labels = build_conditional_latent_cache(vae, dataloader, device="cpu")

    assert all_mu.shape == (N, 4, 8, 8), f"Expected mu shape ({N},4,8,8), got {all_mu.shape}"
    assert all_logvar.shape == (N, 4, 8, 8), f"Expected logvar shape ({N},4,8,8), got {all_logvar.shape}"
    assert all_labels.shape == (N,), f"Expected labels shape ({N},), got {all_labels.shape}"
    print(f"[OK] shape test passed: mu={tuple(all_mu.shape)}, logvar={tuple(all_logvar.shape)}, labels={tuple(all_labels.shape)}")

    assert torch.equal(all_labels, fake_labels), \
        f"Labels do not match input order/values. Expected {fake_labels.tolist()}, got {all_labels.tolist()}"
    print(f"[OK] label correctness test passed: labels preserved in correct order")

    return all_mu, all_logvar, all_labels


def test_conditional_latent_dataset_getitem(all_mu, all_logvar, all_labels):
    ds = ConditionalLatentDataset(all_mu, all_logvar, all_labels)

    z, label = ds[3]
    assert z.shape == (4, 8, 8), f"Expected z shape (4,8,8), got {z.shape}"
    assert label.item() == 3, f"Expected label 3, got {label.item()}"
    print(f"[OK] __getitem__ test passed: z shape={tuple(z.shape)}, label={label.item()}")


def test_conditional_latent_dataset_fresh_sampling_same_label(all_mu, all_logvar, all_labels):
    ds = ConditionalLatentDataset(all_mu, all_logvar, all_labels)

    z1, label1 = ds[5]
    z2, label2 = ds[5]

    assert not torch.allclose(z1, z2), \
        "Two calls to __getitem__(5) returned identical z -- reparameterization may not be resampling."
    assert label1.item() == label2.item(), \
        "Label changed between calls -- label should be fixed, only z should resample."
    print(f"[OK] fresh sampling test passed: z differs between calls, label stays fixed at {label1.item()}")


def run_conditional_latent_dataset_suite():
    print("===== CONDITIONAL LATENT DATASET TEST =====")
    all_mu, all_logvar, all_labels = test_build_conditional_latent_cache_shapes_and_labels()
    test_conditional_latent_dataset_getitem(all_mu, all_logvar, all_labels)
    test_conditional_latent_dataset_fresh_sampling_same_label(all_mu, all_logvar, all_labels)
    print("===== ALL TESTS PASSED =====")


if __name__ == "__main__":
    run_conditional_latent_dataset_suite()