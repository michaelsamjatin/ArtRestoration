import os
from torch.utils.data import Dataset
from PIL import Image

class CrackDataset(Dataset):
    def __init__(self, clean_dir, cracked_dir, mask_dir, transform=None):
        self.clean_dir = clean_dir
        self.cracked_dir = cracked_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get all image files
        self.image_files = [f for f in os.listdir(clean_dir) if f.endswith(('.png', '.jpg'))]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load images
        clean_img = Image.open(os.path.join(self.clean_dir, img_name)).convert('RGB')
        cracked_img = Image.open(os.path.join(self.cracked_dir, img_name)).convert('RGB')
        crack_mask = Image.open(os.path.join(self.mask_dir, img_name)).convert('RGB')
        
        # Apply transforms
        if self.transform:
            clean_img = self.transform(clean_img)
            cracked_img = self.transform(cracked_img)
            crack_mask = self.transform(crack_mask)
        
        return clean_img, cracked_img, crack_mask