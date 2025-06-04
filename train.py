import os
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import utils

from vae import VAE, PerceptualLoss, vae_loss
from preprocess import preprocess_and_split, FaceDataset
# from audioe import AudioEncoder

RAW_IMAGE_DIR = "archive_celeba/"
PREPROCESS_DIR = "archive_preproc_imgs/"
IMAGE_SIZE = (64, 64)
TRAIN_SPLIT = 0.8
BATCH_SIZE = 64
LATENT_DIM = 128
LR = 1e-4
EPOCHS = 20
BETA = 1.0
GAMMA = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.isdir(PREPROCESS_DIR):
    print("→ Preprocessing raw images and splitting into train/test …")
    preprocess_and_split(
        raw_dir=RAW_IMAGE_DIR,
        out_dir=PREPROCESS_DIR,
        image_size=IMAGE_SIZE,
        train_split=TRAIN_SPLIT,
    )
else:
    print(f"Found existing '{PREPROCESS_DIR}/'. Skipping preprocessing.")

# 5.2 Create Datasets & DataLoaders
train_folder = os.path.join(PREPROCESS_DIR, "train")
test_folder = os.path.join(PREPROCESS_DIR, "test")

train_ds = FaceDataset(train_folder)
test_ds = FaceDataset(test_folder)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 5.3 Instantiate Model, Loss, Optimizer
vae = VAE(latent_dim=LATENT_DIM).to(DEVICE)
perceptual_fn = PerceptualLoss(device=DEVICE).to(DEVICE)
optimizer = optim.Adam(vae.parameters(), lr=LR)

# 5.4 Training Loop
vae.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    total_recon = 0.0
    total_perc = 0.0
    total_kl = 0.0

    for imgs in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        imgs = imgs.to(DEVICE)  # (B×3×64×64)

        optimizer.zero_grad()
        recon_imgs, mu, logvar = vae(imgs)

        loss, rl, pl, kl = vae_loss(recon_imgs, imgs, mu, logvar, perceptual_fn, beta=BETA)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_recon += rl.item() * imgs.size(0)
        total_perc += pl.item() * imgs.size(0)
        total_kl += kl.item() * imgs.size(0)

    n_train = len(train_loader.dataset)
    print(
        f"  → Train ‖ "
        f"Loss: {total_loss/n_train:.4f} | "
        f"Recon: {total_recon/n_train:.4f} | "
        f"Perc: {total_perc/n_train:.4f} | "
        f"KL: {total_kl/n_train:.4f}"
    )

    # 5.5 Evaluate on Test Set (every epoch)
    vae.eval()
    with torch.no_grad():
        t_loss = 0.0
        t_recon = 0.0
        t_perc = 0.0
        t_kl = 0.0
        for imgs in tqdm(test_loader, desc=f"Epoch {epoch}/{EPOCHS} [Test]"):
            imgs = imgs.to(DEVICE)
            recon_imgs, mu, logvar = vae(imgs)
            loss, rl, pl, kl = vae_loss(recon_imgs, imgs, mu, logvar, perceptual_fn, beta=BETA)

            t_loss += loss.item() * imgs.size(0)
            t_recon += rl.item() * imgs.size(0)
            t_perc += pl.item() * imgs.size(0)
            t_kl += kl.item() * imgs.size(0)

        n_test = len(test_loader.dataset)
        print(
            f"  → Test  ‖ "
            f"Loss: {t_loss/n_test:.4f} | "
            f"Recon: {t_recon/n_test:.4f} | "
            f"Perc: {t_perc/n_test:.4f} | "
            f"KL: {t_kl/n_test:.4f}"
        )

        # Save one batch of reconstructions vs originals (only once, in epoch 1)
        if epoch == 1:
            sample_imgs = next(iter(test_loader)).to(DEVICE)  # first batch
            recon_imgs, _, _ = vae(sample_imgs)
            # Stack originals on top of reconstructions
            comparison = torch.cat([sample_imgs[:16], recon_imgs[:16]], dim=0)  # (32×3×64×64)
            utils.save_image(
                comparison.cpu(),
                f"recon_comparison_epoch{epoch}.png",
                nrow=16,
            )
            print(f"Saved a recon vs. real comparison → recon_comparison_epoch{epoch}.png")

    vae.train()

print("Training complete. You can inspect the saved `recon_comparison_epoch1.png` to see how well the VAE is doing.")
