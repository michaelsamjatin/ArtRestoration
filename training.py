from datasets import InitDataset, CrackDatasetAug
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from AttentionUNet import AttentionUnet
from train_functions import train_model
import matplotlib.pyplot as plt
import tqdm
from torchmetrics.functional.segmentation import mean_iou
from train_functions import balanced_accuracy, segmentation_loss
from helper import save_image
from torchmetrics.functional.image import structural_similarity_index_measure
import torch.nn.functional as F

working_dir = os.getcwd()

# create data loaders
init_dataset = InitDataset(
    img_dir=r"/content/drive/MyDrive/Colab Notebooks/DLCV/Project/damage_only/damaged_paintings",
    msk_dir=r"/content/drive/MyDrive/Colab Notebooks/DLCV/Project/damage_only/crack_mask",
)

# split the data
train_data, val_data, test_data = torch.utils.data.random_split(init_dataset, [0.65, 0.1, 0.25])

# create the different subsets and only apply augmentation transforms to the training data
train_data = CrackDatasetAug(train_data, apply_transform=True)
val_data = CrackDatasetAug(val_data)
test_data = CrackDatasetAug(test_data)

train_loader = DataLoader(train_data, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_data, batch_size=4, shuffle=False, num_workers=2)

#initialize model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = AttentionUnet(in_channels=3, out_channels=1, filters=[32, 64, 128, 256]).to(device)

train_losses, val_losses = train_model(model, 
                                       train_loader, 
                                       val_loader, 
                                       stop_epoch=100, 
                                       start_epoch=0, 
                                       device=device, 
                                       model_name="Attention_Unet_Aug_l", 
                                       plot_dir="data/output_AugUNet_l", lr=(1e-4))


# create and save a figure with the losses
fig, ax = plt.subplots(figsize=(16,5))

ax.plot(train_losses, label="train")

ax.plot(val_losses, label="validation")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
plt.legend()
plt.savefig(f"Training_progression_AttUNet_l.png")


# Evaluate the model
# create a test data loader
test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=2)

# testing procedure (calcualting the losses, saving predicted masks alongside the damaged painting and true mask)
# forward pass into model
model.eval()

losses = []
ssims = []
balanced_accs = []
mean_ious = []

i = 0

with torch.no_grad():
  for images, msks in tqdm.tqdm(test_loader):
    # transfer to device
    images = images.to(device)
    msks = msks.to(device)

    # compute the forward pass
    learnt_mask = model(images)

    # get MSE and SSIM
    mask_loss = segmentation_loss(learnt_mask, msks)

    threshold = 0.5
    output_binary = (F.sigmoid(learnt_mask)>threshold).long()

    balanced_accs.append(balanced_accuracy(output_binary, msks.long()).item())
    mean_ious.append(mean_iou(output_binary, msks.long(), num_classes=2).mean().item())
    ssim = structural_similarity_index_measure(output_binary.float(), msks).item()

    losses.append(mask_loss.item())
    ssims.append(ssim)

    path = os.path.join(working_dir, "data/output_AugUNet_l")
    save_image(images[0], output_binary[0], msks[0], epoch=-1, step=i, plot_dir=path)
    i += 1

print(f"Mean Loss {np.mean(losses)}")
print(f"Mean SSIM {np.mean(ssims)}")
print(f"Mean balanced accuracy {np.mean(balanced_accs)}")
print(f"Mean IoU {np.mean(mean_ious)}")