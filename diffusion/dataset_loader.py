import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class InpaintingDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        img_files = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.jpg')]

        img_files.sort(key=lambda x: int(x.split('.')[0]))
        mask_files.sort(key=lambda x: int(x.split('.')[0]))

        self.img_paths = [os.path.join(img_dir, f) for f in img_files]
        self.mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Read the image and mask
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = mask / 255.0 
        mask = np.expand_dims(mask, axis=0)
        
        if self.transform:
            img = self.transform(img)
        
        # Return a dictionary with appropriate keys
        return {
            'corrupted_image': torch.tensor(img, dtype=torch.float32),
            'image': torch.tensor(img, dtype=torch.float32),  # Assuming corrupted_image and image are the same
            'mask': torch.tensor(mask, dtype=torch.float32)
        }

# Dataset loading utility
def get_dataset(img_dir, mask_dir):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
    ])
    return InpaintingDataset(img_dir, mask_dir, transform)
