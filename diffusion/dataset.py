import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_folder, image_size=256):
        self.image_folder = image_folder
        self.image_paths = []

        # Traverse the directory and its subdirectories
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                if file.endswith('.jpg'):  # Add any image extensions you need here
                    self.image_paths.append(os.path.join(root, file))
        
        # Define the transformations
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),  # Resize to 256x256
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize between -1 and 1
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Ensure RGB format
        return self.transform(image)

