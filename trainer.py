import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from models.vae import VAE
from diffusion.losses import VAELoss
from utils.logging import Logger


def train_vae(
    dataloader,
    epochs=10,
    lr=1e-4,
    kl_weight=1e-6,
    checkpoint_dir="checkpoints/vae",
    project_name="ldm-vae",
    log_enabled=True,
):
    accelerator = Accelerator(mixed_precision="fp16")

    vae = VAE()
    loss_fn = VAELoss(kl_weight=kl_weight)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    vae, optimizer, dataloader, scheduler = accelerator.prepare(
        vae, optimizer, dataloader, scheduler,
    )

    loss_fn.lpips_loss = loss_fn.lpips_loss.to(accelerator.device)

    logger = Logger(
        project_name=project_name,
        config={"epochs": epochs, "lr": lr, "kl_weight": kl_weight},
        enabled=log_enabled and accelerator.is_main_process,
    )

    global_step = 0
    for epoch in range(epochs):
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch

            optimizer.zero_grad()

            out = vae(x)
            recon, mu, logvar = out["recon"], out["mu"], out["logvar"]

            total_loss, parts = loss_fn(recon, x, mu, logvar)

            accelerator.backward(total_loss)
            accelerator.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()

            if accelerator.is_main_process:
                logger.log_dict(parts, step=global_step)

            global_step += 1

        scheduler.step()

        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{epochs} done. Last loss: {parts['total_loss']:.4f}")
            accelerator.save_state(f"{checkpoint_dir}/epoch_{epoch+1}")

    if accelerator.is_main_process:
        unwrapped_vae = accelerator.unwrap_model(vae)
        torch.save(unwrapped_vae.state_dict(), f"{checkpoint_dir}/vae_final.pt")
        print(f"Final VAE weights saved to {checkpoint_dir}/vae_final.pt")

    logger.finish()

    return vae


if __name__ == "__main__":
    import argparse
    from torch.utils.data import TensorDataset, DataLoader
    from utils.config import load_config
    from dataset import get_cifar10_datasets, ImageOnlyDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/vae_config.yaml")
    parser.add_argument("--smoketest", action="store_true")
    args = parser.parse_args()

    if args.smoketest:
        dummy_data = torch.randn(16, 3, 32, 32)
        dummy_dataset = TensorDataset(dummy_data)
        dummy_dataloader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)

        train_vae(
            dataloader=dummy_dataloader,
            epochs=1,
            checkpoint_dir="checkpoints/vae_smoketest",
            project_name="ldm-vae-smoketest",
            log_enabled=False,
        )
        print("Smoke test complete.")
    else:
        cfg = load_config(args.config)

        train_set, val_set, test_set = get_cifar10_datasets(
            root=cfg["data"]["root"],
            train_val_split=cfg["data"]["train_val_split"],
        )
        wrapped_train = ImageOnlyDataset(train_set)
        train_dataloader = DataLoader(
            wrapped_train, batch_size=cfg["training"]["batch_size"], shuffle=True
        )

        train_vae(
            dataloader=train_dataloader,
            epochs=cfg["training"]["epochs"],
            lr=cfg["training"]["lr"],
            kl_weight=cfg["training"]["kl_weight"],
            checkpoint_dir=cfg["checkpointing"]["checkpoint_dir"],
            project_name=cfg["logging"]["project_name"],
            log_enabled=cfg["logging"]["log_enabled"],
        )