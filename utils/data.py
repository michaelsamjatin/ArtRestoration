import os
import torch
from torch.utils.data import Dataset, random_split
import torchvision.transforms as T
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

class InpaintingDataset(Dataset):
    """
    Implements a custom PyTorch dataset to load images and masks used for inpainting.
    """
    def __init__(self, original_dir, damaged_dir, transform=None):
        """
        Args:
            original_dir (str): Path to the folder with original paintings.
            damaged_dir (str): Path to the folder with damaged paintings.
            transform (callable, optional): Transform to be applied to images.

        Raises:
            AssertionError: If there is a mismatch in the number of images across folders.
        """
        self.original_dir = original_dir
        self.damaged_dir = damaged_dir
        self.transform = transform
        
        # Helper function to filter valid image files
        def valid_image_files(folder1, folder2):
            return sorted([
                file for file in os.listdir(folder1)
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and os.path.exists(folder2 / file)
            ])

        self.original_images = valid_image_files(original_dir, damaged_dir)
        self.damaged_images = valid_image_files(damaged_dir, original_dir)

    def __len__(self):
        return len(self.original_images)

    def __getitem__(self, idx):
        # Load images
        original_path = os.path.join(self.original_dir, self.original_images[idx])
        damaged_path = os.path.join(self.damaged_dir, self.damaged_images[idx])

        original = Image.open(original_path).convert("RGB")
        damaged = Image.open(damaged_path).convert("RGB")

        # Apply speficied transformations
        if self.transform:
            original = self.transform(original)
            mask = self.transform(mask)
            damaged = self.transform(damaged)

        return damaged, mask, original


def get_dataset(root: str, transform, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2):
    """
    Fetches inpainting dataset and performs split into training, validation, and test sets.

    Args:
        root (str): Path to the root directory containing the dataset folders:
            - `data/initial`: Original images.
            - `data/crack_mask`: Masks representing damaged areas.
            - `data/damaged_paintings`: Images with applied damage.
        transform (callable): Transformations to be applied to the dataset samples.
        train_ratio (float, optional): Proportion of the dataset to use for training. Defaults to 0.7.
        val_ratio (float, optional): Proportion of the dataset to use for validation. Defaults to 0.1.
        test_ratio (float, optional): Proportion of the dataset to use for testing. Defaults to 0.2.

    Returns:
        tuple: A tuple containing three subsets of the dataset:
            - `data_train` (torch.utils.data.Subset): Training subset.
            - `data_val` (torch.utils.data.Subset): Validation subset.
            - `data_test` (torch.utils.data.Subset): Test subset.

    Raises:
        AssertionError: If the sum of `train_ratio`, `val_ratio`, and `test_ratio` does not equal 1.0.
    """

    # Define paths to data
    original_dir = root / 'initial'
    damaged_dir = root / 'damaged_paintings'

    # Init Inpainting Dataset
    dataset = InpaintingDataset(original_dir, damaged_dir, transform)

    # Ensure the proportions sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1."
        
    # Calculate dataset sizes
    num_samples = len(dataset)
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size
        
    # Split the dataset
    data_train, data_val, data_test = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))
    
    return data_train, data_val, data_test
