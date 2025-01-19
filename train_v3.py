import os
import torch
import numpy as np
import tqdm
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from model_v3 import UNet
from crack_data import CrackDataset
from torchmetrics.functional.classification import specificity, recall, dice
from torchmetrics.functional.segmentation import mean_iou
from torchmetrics.functional.image import structural_similarity_index_measure
from torchvision.ops import sigmoid_focal_loss
from pathlib import Path
from PIL import Image

def balanced_accuracy(pred, targets):
  specificity_val = specificity(pred, targets, task="binary")
  recall_val = recall(pred, targets, task="binary")
  return (recall_val + specificity_val)/2

def train_step(model, optimizer, cracked_image, crack_mask):
  optimizer.zero_grad()
  # forward pass
  learnt_mask = model(cracked_image)

  # combination of SSIM and MSE loss
  #mask_loss = (1 - ssim(learnt_mask, crack_mask) + F.mse_loss(learnt_mask, crack_mask))
  bce = F.cross_entropy(learnt_mask, crack_mask)
  focal_loss = sigmoid_focal_loss(learnt_mask, crack_mask, reduction='mean', alpha=0.5)
  dice_loss = (1-dice(learnt_mask, crack_mask.long()))
  mask_loss = bce + focal_loss + dice_loss

  total_loss = mask_loss

  # Backward pass
  total_loss.backward()
  optimizer.step()

  return total_loss, learnt_mask


def train_model(model, train_loader, val_loader, num_epochs, device):
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

  best_val_loss = float('inf')

  for epoch in range(num_epochs):
    model.train()
    train_losses = []
    train_bal_accs = []
    train_mean_ious = []

    for cracked_imgs, crack_masks in tqdm.tqdm(train_loader):

      cracked_imgs = cracked_imgs.to(device)
      crack_masks = crack_masks.to(device)

      # training step
      loss, learnt_mask = train_step(model, optimizer, cracked_imgs, crack_masks)
      train_losses.append(loss.item())

      # convert the probabilistic output to 0/1 (as integers not floats!)
      threshold = 0.5
      output_binary = (learnt_mask>threshold).long()
      crack_masks = crack_masks.long()

      # calculate the training accuracies / mIoU
      train_bal_accs.append(balanced_accuracy(output_binary, crack_masks).item())
      train_mean_ious.append(mean_iou(output_binary, crack_masks, num_classes=2).mean().item())

      # validation
      model.eval()
      val_losses = []
      val_bal_accs = []
      val_mean_ious = []

      with torch.no_grad():
        for cracked_imgs, crack_masks in val_loader:
          cracked_imgs = cracked_imgs.to(device)
          crack_masks = crack_masks.to(device)

          learnt_mask = model(cracked_imgs)
          val_loss = F.mse_loss(learnt_mask, crack_masks)
          val_losses.append(val_loss.item())

          # convert probabilistic output to binary
          threshold = 0.5
          output_binary = (learnt_mask>threshold).long()
          crack_masks = crack_masks.long()

          # calcaulate the val accuracies / mIoU
          val_bal_accs.append(balanced_accuracy(output_binary, crack_masks).item())
          val_mean_ious.append(mean_iou(output_binary, crack_masks, num_classes=2).mean().item())

      avg_val_loss = np.mean(val_losses)
      avg_train_bal_acc = np.mean(train_bal_accs)
      avg_train_m_ious = np.mean(train_mean_ious)

      avg_val_bal_acc = np.mean(val_bal_accs)
      avg_val_m_ious = np.mean(val_mean_ious)

      print(f"Epoch {epoch}: Avg Train Loss: {np.mean(train_losses):.4f}, Avg Train Bal Acc: {avg_train_bal_acc:.4f}, Avg Mean IoU: {avg_train_m_ious:.4f}")
      print(f"Epoch {epoch}: Avg Val Loss: {avg_val_loss:.4f}, Avg Val Bal Acc: {avg_val_bal_acc:.4f}, Avg Mean IoU: {avg_val_m_ious:.4f}")

if __name__ == '__main__':

    # Setup transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create datasets
    dataset = CrackDataset(
        img_dir=r"C:\Users\larao\OneDrive\Desktop\Master\WiSe2425\DLCV\Projekt\ArtRestoration\only_cracks\damaged_paintings",
        msk_dir=r"C:\Users\larao\OneDrive\Desktop\Master\WiSe2425\DLCV\Projekt\ArtRestoration\only_cracks\crack_mask",
        transform=transform
    )

    # split data into train and test
    train_split = 0.7
    train_size = np.floor(dataset.__len__() * train_split)
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, dataset.__len__()-train_size])

    val_split = 0.1
    val_size = np.floor(train_data.__len__() * val_split)
    train_data, val_data = torch.utils.data.random_split(train_data, [train_data.__len__()-val_size, val_size])

    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=2)

    # Initialize model and move to device
    # MPS for accelerated training on Apple Silicon, change to cuda if using HLR
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)

    # Train model
    train_model(model, train_loader, val_loader, num_epochs=50, device=device)

    ############ saving the model? #################


    # testing the model?
    model.eval()

    mses = []
    ssims = []
    balanced_accs = []
    mean_ious = []

    with torch.no_grad():
      for i, (images, msks) in enumerate(test_loader):
        # transfer to device
        images = images.to(device)
        msks = msks.to(device)

        # compute the forward pass
        learnt_masks = model(images)

        # get MSE and SSIM
        mse = F.mse_loss(learnt_masks, msks)

        threshold = 0.5
        output_binary = (learnt_masks>threshold).long()

        balanced_accs.append(balanced_accuracy(output_binary, msks.long()).item())
        mean_ious.append(mean_iou(output_binary, msks.long(), num_classes=2).mean().item())
        ssim = structural_similarity_index_measure(output_binary.float(), msks).item()
        
        mses.append(mse.item())
        ssims.append(ssim)

        # save the learnt mask of first image in each batch
        prediction = (output_binary[0] * 255).permute(1,2,0).cpu()

        NEW_DIR = r"C:\Users\larao\OneDrive\Desktop\Master\WiSe2425\DLCV\Projekt\ArtRestoration\predicted_masks"
        new_directory = Path(NEW_DIR)

        f = f"sample_prediction_{i}.jpg"

        path = os.path.join(NEW_DIR, f)
        new_directory.mkdir(exist_ok=True)

        im = Image.fromarray(prediction)
        im.save(path)


    print(f"Mean MSE {np.mean(mses)}")
    print(f"Mean SSIM {np.mean(ssims)}")
    print(f"Mean balanced accuracy {np.mean(balanced_accs)}")
    print(f"Mean IoU {np.mean(mean_ious)}")

