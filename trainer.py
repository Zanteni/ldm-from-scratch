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
    # TODO: build accelerator = Accelerator(mixed_precision="fp16")
    accelerator = Accelerator(mixed_precision="fp16")

    # TODO: build vae = VAE()
    vae = VAE()

    # TODO: build loss_fn = VAELoss(kl_weight=kl_weight)
    loss_fn = VAELoss(kl_weight=kl_weight)

    # TODO: build optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
    optimizer = torch.optim.AdamW(vae.parameters(),lr=lr)

    # TODO: build scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)
    # TODO: prepare vae, optimizer, dataloader, scheduler through accelerator.prepare(...)
    #       (assign back to the same variable names)
    vae, optimizer, dataloader, scheduler = accelerator.prepare(
    vae,
    optimizer,
    dataloader,
    scheduler,
    )
    # TODO: move loss_fn.lpips_loss to accelerator.device
    loss_fn.lpips_loss = loss_fn.lpips_loss.to(accelerator.device)

    # TODO: build logger = Logger(project_name=..., config={...}, enabled=log_enabled and accelerator.is_main_process)
    logger = Logger(project_name=project_name,
                    config={"epochs":epochs,"lr":lr,"kl_weight":kl_weight},
                    enabled=log_enabled and accelerator.is_main_process)
    global_step = 0
    for epoch in range(epochs):
        for batch in dataloader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch

            # TODO: optimizer.zero_grad()
            optimizer.zero_grad()

            # TODO: forward pass -- out = vae(x), extract recon, mu, logvar
            out = vae(x)
            recon, mu, logvar = out["recon"],out["mu"],out["logvar"]
            # TODO: compute total_loss, parts = loss_fn(recon, x, mu, logvar)
            total_loss,parts = loss_fn(recon,x,mu,logvar)
            # TODO: accelerator.backward(total_loss)
            accelerator.backward(total_loss)

            # TODO: accelerator.clip_grad_norm_(vae.parameters(), 1.0)
            accelerator.clip_grad_norm_(vae.parameters(),1.0)
            # TODO: optimizer.step()
            optimizer.step()

            # TODO: if accelerator.is_main_process: logger.log_dict(parts, step=global_step)
            if  accelerator.is_main_process:
                logger.log_dict(parts,step=global_step)

            global_step += 1

        # TODO: scheduler.step()  -- once per epoch
        scheduler.step()

        # TODO: if accelerator.is_main_process:
        #           print epoch summary (e.g. f"Epoch {epoch+1}/{epochs} done. Last loss: {parts['total_loss']:.4f}")
        #           accelerator.save_state(f"{checkpoint_dir}/epoch_{epoch+1}")
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}/{epochs} done. Last loss: {parts['total_loss']:.4f}")
            accelerator.save_state(f"{checkpoint_dir}/epoch_{epoch+1}")

    # TODO: logger.finish()
    logger.finish()

    return vae


if __name__ == "__main__":
    from torch.utils.data import TensorDataset, DataLoader

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