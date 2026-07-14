import torch
import torch.nn as nn
import lpips

from models.vae import VAE


class VAELoss(nn.Module):
    def __init__(self, kl_weight=1e-6, lpips_net="vgg"):
        super().__init__()
        self.kl_weight = kl_weight
        self.lpips_loss = lpips.LPIPS(net=lpips_net)

        for p in self.lpips_loss.parameters():
            p.requires_grad = False

    def forward(self, recon, target, mu, logvar):
        perceptual = self.lpips_loss(recon, target).mean()
        kl_loss = VAE.kl_divergence(mu, logvar)
        total = perceptual + self.kl_weight * kl_loss

        return total, {
            "total_loss": total.item(),
            "perceptual_loss": perceptual.item(),
            "kl_loss": kl_loss.item(),
        }