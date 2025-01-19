import os
from torch.utils.data import Dataset
from PIL import Image

class CrackDataset(Dataset):
  def __init__(self, img_dir, msk_dir, transform=None):
    self.img_dir = img_dir
    self.msk_dir = msk_dir
    self.transform = transform

    # get the image files
    self.image_files = [f for f in os.listdir(img_dir) if f.endswith('jpg')]

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    img_name = self.image_files[idx]

    # load images
    img = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
    # convert the mask to a bw image (just having 0s and 1s)
    msk = Image.open(os.path.join(self.msk_dir, img_name)).convert("1")


    # apply transforms
    if self.transform:
      img = self.transform(img)
      msk = self.transform(msk)

    return img, msk