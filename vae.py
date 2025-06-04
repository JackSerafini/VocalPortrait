import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

class VAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: input 3x64x64 -> latent mean and log variance
        self.encoder = nn.Sequential(
            # Layer 1: 3x64x64 -> 64x32x32 # Should I use bigger images?
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), # inplace=True?

            # Layer 2: 64x32x32 -> 128x16x16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Layer 3: 128x16x16 -> 256x8x8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            # Layer 4: 256x8x8 -> 512x4x4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            # Layer 1: 512*4*4 -> 256*8*8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(), # inplace=True?

            # Layer 2: 256*8*8 -> 128*16*16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Layer 3: 128*16*16 -> 64*32*32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 4: 64*32*32 -> 3*64*64
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterization(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(h.size(0), 512, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterization(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar
    
class PerceptualLoss(nn.Module):
    def __init__(self, device = 'cuda'):
        super(PerceptualLoss, self).__init__()
        self.device = torch.device(device)

        vgg = models.vgg19(pretrained = True).features.to(self.device).eval()
        self.layers = [5, 15, 25]
        self.vgg = nn.ModuleList([vgg[i] for i in self.layers]).to(self.device)
        
        for param in vgg.parameters():
            param.requires_grad = False

    def forward(self, real_x, recon_x):
        loss = 0.0
        for layer in self.vgg:
            f_recon = layer(recon_x)
            f_real = layer(real_x).detach()
            loss += F.l1_loss(f_recon, f_real)
        return loss / len(self.vgg)
    
def vae_loss(recon_x, x, mu, logvar, perceptual_loss_fn, beta: int = 1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

    perceptual = perceptual_loss_fn(x, recon_x)

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + perceptual + beta * kl, recon_loss, perceptual, kl