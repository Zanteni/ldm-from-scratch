import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from models.latent_unet import LatentUNet, ConditionalLatentUNet
from diffusion.scheduler import get_beta_schedule, get_diffusion_constants
from diffusion.diffusion_loss import compute_diffusion_loss
from utils.logging import Logger


def train_diffusion(
    dataloader,
    T=1000,
    schedule_type="linear",
    epochs=10,
    lr=1e-4,
    z_channels=4,
    base_channels=128,
    time_emb_dim=512,
    N=6,
    L=4,
    max_mult=4,
    checkpoint_dir="checkpoints/diffusion",
    project_name="ldm-diffusion",
    log_enabled=True,
    resume_checkpoint=None,
    resume_from_epoch=0,
    conditional=False,
    num_classes=10,
    class_emb_dim=256,
    drop_prob=0.1,
):
    accelerator = Accelerator(mixed_precision="fp16")

    if conditional:
        model = ConditionalLatentUNet(
            z_channels=z_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
            num_classes=num_classes,
            class_emb_dim=class_emb_dim,
            N=N,
            L=L,
            max_mult=max_mult,
        )
    else:
        model = LatentUNet(
            z_channels=z_channels,
            base_channels=base_channels,
            time_emb_dim=time_emb_dim,
            N=N,
            L=L,
            max_mult=max_mult,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_epochs = resume_from_epoch + epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=total_epochs)

    betas = get_beta_schedule(schedule_type=schedule_type, timesteps=T)
    constants = get_diffusion_constants(betas=betas)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    constants["sqrt_alphas_cumprod"] = constants["sqrt_alphas_cumprod"].to(accelerator.device)
    constants["sqrt_one_minus_alphas_cumprod"] = constants["sqrt_one_minus_alphas_cumprod"].to(accelerator.device)

    if resume_checkpoint is not None:
        accelerator.load_state(resume_checkpoint)
        scheduler.T_max = total_epochs
        if accelerator.is_main_process:
            print(f"Resumed from {resume_checkpoint}, continuing from epoch {resume_from_epoch}")

    config = {
        "epochs": total_epochs,
        "lr": lr,
        "T": T,
        "schedule_type": schedule_type,
        "z_channels": z_channels,
        "base_channels": base_channels,
        "time_emb_dim": time_emb_dim,
        "N": N,
        "L": L,
        "resumed_from": resume_from_epoch,
        "conditional": conditional,
    }
    if conditional:
        config.update({
            "num_classes": num_classes,
            "class_emb_dim": class_emb_dim,
            "drop_prob": drop_prob,
        })

    logger = Logger(
        project_name=project_name,
        config=config,
        enabled=log_enabled and accelerator.is_main_process,
    )

    global_step = 0
    for epoch in range(resume_from_epoch, total_epochs):
        for batch in dataloader:
            if conditional:
                x, labels = batch
            else:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                labels = None

            B = x.shape[0]
            t = torch.randint(0, T, (B,), device=accelerator.device)

            if conditional:
                drop_mask = torch.rand(B, device=accelerator.device) < drop_prob
                null_idx = num_classes
                labels = torch.where(drop_mask, torch.full_like(labels, null_idx), labels)

            optimizer.zero_grad()
            loss = compute_diffusion_loss(
                model, x, t,
                constants["sqrt_alphas_cumprod"],
                constants["sqrt_one_minus_alphas_cumprod"],
                class_labels=labels,
            )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            global_step += 1

            if accelerator.is_main_process:
                logger.log_dict({"loss": loss.item()}, step=global_step)

        scheduler.step()
        if accelerator.is_main_process:
            print(f"epoch {epoch+1}/{total_epochs} done. Last loss: {loss.item():.4f}")
            accelerator.save_state(f"{checkpoint_dir}/epoch_{epoch+1}")

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), f"{checkpoint_dir}/latent_final.pt")
        print(f"Final diffusion model weights saved to {checkpoint_dir}/latent_final.pt")

    logger.finish()
    return model


if __name__ == "__main__":
    from torch.utils.data import TensorDataset

    # --- Unconditional smoke test ---
    dummy_z = torch.randn(16, 4, 8, 8)
    dummy_dataset = TensorDataset(dummy_z)
    dummy_dataloader = DataLoader(dummy_dataset, batch_size=8, shuffle=True)

    train_diffusion(
        dataloader=dummy_dataloader,
        T=100,
        epochs=1,
        N=4,
        L=2,
        checkpoint_dir="checkpoints/diffusion_smoketest",
        project_name="ldm-diffusion-smoketest",
        log_enabled=False,
    )
    print("Unconditional smoke test complete.")

    # --- Conditional smoke test ---
    dummy_z_cond = torch.randn(16, 4, 8, 8)
    dummy_labels = torch.randint(0, 10, (16,))
    dummy_conditional_dataset = TensorDataset(dummy_z_cond, dummy_labels)
    dummy_conditional_dataloader = DataLoader(dummy_conditional_dataset, batch_size=8, shuffle=True)

    train_diffusion(
        dataloader=dummy_conditional_dataloader,
        T=100,
        epochs=1,
        N=4,
        L=2,
        conditional=True,
        num_classes=10,
        checkpoint_dir="checkpoints/diffusion_conditional_smoketest",
        project_name="ldm-diffusion-conditional-smoketest",
        log_enabled=False,
    )
    print("Conditional smoke test complete.")