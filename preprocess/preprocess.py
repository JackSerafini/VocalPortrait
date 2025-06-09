import os
import glob
import random
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset

def preprocess_and_split(raw_dir: str, out_dir: str, image_size: tuple = (64, 64), train_split: float = 0.8,):
    """
    1. Reads all JPG/PNG/etc. from raw_dir.
    2. Tries to open each with PIL; if loadable, resizes to image_size and saves as PNG.
    3. Shuffles and splits into train/ and test/ (train_split fraction).
    """

    raw_paths = glob(os.path.join(raw_dir, "*"))
    random.shuffle(raw_paths)

    n_total = len(raw_paths)
    n_train = int(train_split * n_total)

    train_paths = raw_paths[:n_train]
    test_paths = raw_paths[n_train:]

    train_out = os.path.join(out_dir, "train")
    test_out = os.path.join(out_dir, "test")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    def _process_list(paths, destination):
        for p in tqdm(paths, desc=f"Processing → {destination}"):
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                continue  # skip any unreadable file
            img = img.resize(image_size, resample=Image.BILINEAR)
            fname = Path(p).stem + ".png"
            save_path = os.path.join(destination, fname)
            img.save(save_path)  # saved as PNG in [0,255]
    
    _process_list(train_paths, train_out)
    _process_list(test_paths, test_out)


class FaceDataset(Dataset):
    """
    Loads preprocessed PNGs from a folder (train/ or test/),
    converts to float32 tensor in [0,1] with shape (3xHxW).
    """
    def __init__(self, folder: str, transform=None):
        super().__init__()
        self.files = sorted(glob(os.path.join(folder, "*.png")))
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),  # PIL [0–255] → float [0–1], shape (3,H,W)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)