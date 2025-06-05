import torch
from torchvision.utils import save_image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import os

from plain_vae import plainVAE

import random
from torchvision.transforms import functional as F

class RandomBrightnessTransform:
    def __init__(self, brightness_range=(0.5, 1.5)):
        self.brightness_range = brightness_range

    def __call__(self, img):
        # Randomly choose a brightness factor
        brightness_factor = random.uniform(*self.brightness_range)
        return F.adjust_brightness(img, brightness_factor)

# ----------------------------
# 1. Settings
# ----------------------------
checkpoint_path = "epoch20.pth"
output_dir = "vae_test_outputs"
os.makedirs(output_dir, exist_ok=True)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
latent_dim = 128
num_samples = 10

# ----------------------------
# 2. Load Model
# ----------------------------
model = plainVAE(latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print(f"Loaded model from {checkpoint_path}")

# ----------------------------
# 3. Generate New Images
# ----------------------------
with torch.no_grad():
    z = torch.randn(num_samples, latent_dim).to(device)
    generated_imgs = model.decode(z)
    save_image(generated_imgs.cpu(), os.path.join(output_dir, "generated_samples.png"),
               nrow=8, normalize=True)
    print(f"Saved {num_samples} generated images to 'generated_samples.png'")

# ----------------------------
# 4. Optional: Visualize Reconstructions
# ----------------------------

# Prepare dataset
transform = transforms.Compose([
    # RandomBrightnessTransform(),  # Random brightness
    transforms.Resize((128, 128)),  # ensure images are 64x64
    transforms.ToTensor(),  # values ∈ [0, 1],
])

data_root = "cartella"  # same path as during training
dataset = datasets.ImageFolder(root=data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=1)

# Get a single batch of real images
real_imgs, _ = next(iter(dataloader))
real_imgs = real_imgs.to(device)

with torch.no_grad():
    recon_imgs, _, _ = model(real_imgs)

    # Save original and reconstructed side-by-side
    save_image(real_imgs.cpu(), os.path.join(output_dir, "real_images.png"), nrow=1, normalize=True)
    save_image(recon_imgs.cpu(), os.path.join(output_dir, "reconstructed_images.png"), nrow=1, normalize=True)
    print("Saved real and reconstructed images.")
