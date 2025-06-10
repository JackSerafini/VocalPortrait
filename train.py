import os
import random
import torch
from torch import optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from PIL import Image

from models.plain_vae import plainVAE

DATA_ROOT = "archive_celebacrop"
LATENT_DIMENSION = 128
EPOCHS = 40

class VocalPortraitDataset():
    # TODO: labels not implemented
    def __init__(self, root_path:str, transform:None):
        # root_path is the faces folder path
        self.root_path = root_path
        self.transform = transform
        self.image_paths = self.get_image_paths()

    def get_image_paths(self):
        image_paths = []
        
        person_folder = os.listdir(self.root_path)
        for person in person_folder:
            for nationality in ['English', 'Urdu']:
                if os.path.exists(f"{self.root_path}/{person}/{nationality}"):
                    video_ids = os.listdir(f"{self.root_path}/{person}/{nationality}")
                    for video_id in video_ids:
                        image_names = os.listdir(f"{self.root_path}/{person}/{nationality}/{video_id}")
                        for image_name in image_names:
                            image_paths.append(f"{self.root_path}/{person}/{nationality}/{video_id}/{image_name}")
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        # label = self.labels[idx]
        return image, idx

# Typical transforms for a VAE with Sigmoid output:
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),         # this ensures values ∈ [0,1]
])

celeb_dataset = ImageFolder(root = "archive_celebacrop", transform = transform)
ffqh_dataset = ImageFolder(root = "archive_ffqh", transform = transform)
v1_dataset = VocalPortraitDataset(root_path = "archive_vocalcrop/mavceleb_v1_train_cropped/faces", transform = transform)
v2_dataset = VocalPortraitDataset(root_path = "archive_vocalcrop/mavceleb_v2_train_cropped/faces", transform = transform)

def get_random_subset(dataset, n):
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    return Subset(dataset, indices)

celeb_subset = get_random_subset(celeb_dataset, 50_000)
ffqh_subset  = get_random_subset(ffqh_dataset, 50_000)
v1_subset    = get_random_subset(v1_dataset, 25_000)
v2_subset    = get_random_subset(v2_dataset, 25_000)

# Step 4: Combine into a single dataset
combined_dataset = ConcatDataset([celeb_subset, ffqh_subset, v1_subset, v2_subset])

dataloader = DataLoader(
    combined_dataset,
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

resume = True
checkpoint_path = "vae_epoch20.pth" # or whichever epoch you want

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

    if epoch % 5 == 0 or epoch == EPOCHS:
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