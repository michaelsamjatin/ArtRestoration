from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF
import random

class InitDataset(Dataset):
  # dataset class that loads images and masks from their respecitive folders and converts them to tensors
  def __init__(self, img_dir, msk_dir):
    self.img_dir = img_dir
    self.msk_dir = msk_dir

    # get the image files
    self.image_files = [f for f in os.listdir(img_dir) if f.endswith('jpg')]

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    img_name = self.image_files[idx]

    # load images
    img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
    # load masks (and convert to bw)
    msk = Image.open(os.path.join(self.msk_dir, img_name)).convert('1')

    # transform to tensor
    img = TF.to_tensor(img)
    msk = TF.to_tensor(msk)

    return img, msk, img_name
  

class CrackDatasetAug(Dataset):
  # dataset class that gets a dataset (subset) and applies some data augmentation transformation randomly on some of the images
  def __init__(self, subset, apply_transform=False, return_names=False):
    self.subset = subset
    self.apply_transform = apply_transform
    self.return_names = return_names

  def __len__(self):
    return len(self.subset)

  def transform(self, image, mask):
    # all transforms from the functional interface work on PIL images and tensors
    # Random horizontal flipping
    if random.random() > 0.5:
      image = TF.hflip(image)
      mask = TF.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
      image = TF.vflip(image)
      mask = TF.vflip(mask)

    # Color jitter? -> randomly change hue, saturation, contrast, brightness
    # I'll start with contrast, maybe lowering it will force the model to pay more attention to the low-contrast damage
    if random.random() > 0.5:
      image = TF.adjust_contrast(image, contrast_factor=0.5)

    return image, mask

  def __getitem__(self, idx):
    img, msk, img_name = self.subset[idx] #### or do we have to call the getitem function???

    if self.apply_transform:
      img, msk = self.transform(img, msk)

    if self.return_names:
      return img, msk, img_name

    return img, msk
  
class EvaluationDataset(Dataset):
  # dataset class mainly used for evaluation on the real damage images
  def __init__(self, img_dir, return_names=True):
    self.img_dir = img_dir
    self.return_names = return_names

    # get the image files
    self.image_files = [f for f in os.listdir(img_dir) if f.endswith('jpg')]

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    img_name = self.image_files[idx]

    # load images
    img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")

    # Transform to Tensor
    img = TF.to_tensor(img)

    if self.return_names:
      return img, img_name

    return img