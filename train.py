import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder

from plain_vae import plainVAE

DATA_ROOT = "archive"
LATENT_DIMENSION = 128
EPOCHS = 100

# Typical transforms for a VAE with Sigmoid output:
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),         # this ensures values ∈ [0,1]
])

dataset = ImageFolder(root = DATA_ROOT, transform = transform)

dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    pin_memory=True,
)

# 3) ---------- Instantiate model, optimizer, device ----------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device("mps")
model = plainVAE(latent_dim = LATENT_DIMENSION).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 4) ---------- Training loop ----------
log_interval = 100   # how many batches between printouts

resume = False
checkpoint_path = "vae_epoch2.pth" # or whichever epoch you want

start_epoch = 1
if resume and os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Resumed model from {checkpoint_path}")
    # If you also saved the optimizer state, load it too:
    # optimizer.load_state_dict(torch.load("optimizer_epoch5.pth"))
    start_epoch = int(checkpoint_path.split("epoch")[1].split(".")[0]) + 1

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    running_recon_loss = 0.0
    running_kl_loss = 0.0
    running_total_loss = 0.0

    for batch_idx, (imgs, _) in enumerate(dataloader, start=1):
        imgs = imgs.to(device)   # shape (B, 3, 128, 128)
        optimizer.zero_grad()

        recon_imgs, mu, logvar = model(imgs)
        loss, recon_loss, kl_loss = model.loss(imgs, recon_imgs, mu, logvar, beta=1.0)
        loss.backward()
        optimizer.step()

        running_total_loss += loss.item()
        running_recon_loss += recon_loss.item()
        running_kl_loss += kl_loss.item()

        if batch_idx % log_interval == 0:
            avg_total = running_total_loss / log_interval
            avg_recon = running_recon_loss / log_interval
            avg_kl    = running_kl_loss / log_interval
            print(f"Epoch [{epoch}/{EPOCHS}]  "
                  f"Batch [{batch_idx}/{len(dataloader)}]  "
                  f"Total Loss: {avg_total:.3f}  "
                  f"Reconstruction: {avg_recon:.3f}  "
                  f"KL: {avg_kl:.3f}")
            running_total_loss = 0.0
            running_recon_loss = 0.0
            running_kl_loss    = 0.0

    if epoch % 10 == 0 or epoch == EPOCHS:
        checkpoint_path = f"vae_epoch{epoch}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        # (Optional) Also: generate a few samples from N(0,I) and save to disk
        model.eval()
        with torch.no_grad():
            sample_z = torch.randn(64, model.latent_dim).to(device)
            sample_imgs = model.decode(sample_z)   # (64, 3, 128, 128)
            # Save a grid of 64 samples as a single image
            save_image(sample_imgs.cpu(), f"sample_epoch{epoch}.png", nrow=8, normalize=True)

print("Training finished.")