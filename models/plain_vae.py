import torch
from torch import nn
from torch.nn import functional as F

class plainVAE(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super(plainVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: input 3x128x128 -> latent mean and log variance
        self.encoder = nn.Sequential(
            # Layer 1: 3x128x128 -> 32x64x64 # Should I use bigger images?
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # Layer 2: 32x64x64 -> 64x32x32 # Should I use bigger images?
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 3: 64x32x32 -> 128x16x16
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Layer 4: 128x16x16 -> 256x8x8
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # # Layer 5: 256x8x8 -> 512x4x4
            # nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
        )

        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder = nn.Sequential(
            # Layer 1: 512*4*4 -> 256*8*8
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Layer 2: 256*8*8 -> 128*16*16
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Layer 3: 128*16*16 -> 64*32*32
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # # Layer 4: 64*32*32 -> 32*64*64
            # nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),

            # Layer 4: 32*64*64 -> 3*128*128
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        mu = F.normalize(mu, p=2, dim=1)
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

    def loss(self, x, recon_x, mu, logvar, beta: float = 1.0):
        recon_loss = F.mse_loss(x, recon_x, reduction='sum') / x.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

        return recon_loss + beta * kl, recon_loss, kl