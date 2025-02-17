from torchvision.ops import sigmoid_focal_loss
import torch.nn.functional as F
from torchmetrics.functional.classification import specificity, recall, dice
import os
import torch
import numpy as np
import tqdm
from torchmetrics.functional.segmentation import mean_iou
from helper import save_image

# get the working dir
working_dir = os.getcwd()


def segmentation_loss(prediction, truth, weights=[1,1,1], alpha=0.5):
  '''
  calculates the training and validation loss for segmentation by summing binary cross entropy, focal loss and dice loss
  -----
  Parameters:
  prediction: the models raw predictions (aka logits)
  truth: the true labels
  weights: how the three losses are weighted (order: bce, focal, dice)
  alpha: alpha parameter for the focal loss
  -----
  returns:
  the loss
  '''
  bce = F.binary_cross_entropy_with_logits(prediction, truth)
  focal = sigmoid_focal_loss(prediction, truth, reduction='mean', alpha=alpha)
  dice_loss = (1-dice(prediction, truth.long()))

  return (weights[0] * bce) + (weights[1] * focal) + (weights[2] * dice_loss)


def train_step(model, optimizer, cracked_image, crack_mask):
  optimizer.zero_grad()
  # forward pass
  learnt_mask = model(cracked_image)

  total_loss = segmentation_loss(learnt_mask, crack_mask)

  # Backward pass
  total_loss.backward()
  optimizer.step()

  return total_loss, learnt_mask


def balanced_accuracy(pred, targets):
  specificity_val = specificity(pred, targets, task="binary")
  recall_val = recall(pred, targets, task="binary")
  return (recall_val + specificity_val)/2


def train_model(model, train_loader, val_loader, device, stop_epoch, start_epoch=0, lr=1e-4, plot_dir="data/output", model_name="best_model"):
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  scheduler = torch.optim.lr_scheduler(optimizer, step_size=10, gamma=0.7)

  best_val_loss = float('inf')
  val_loss_per_epoch = []
  train_loss_per_epoch = []

  eps_without_improvement = 0

  for epoch in range(start_epoch, stop_epoch):
    model.train()
    train_losses = []
    train_bal_accs = []
    train_mean_ious = []

    os.makedirs(os.path.join(working_dir, plot_dir), exist_ok=True)
    i = 0

    for cracked_imgs, crack_masks in tqdm.tqdm(train_loader):

      cracked_imgs = cracked_imgs.to(device)
      crack_masks = crack_masks.to(device)

      # training step
      loss, learnt_mask = train_step(model, optimizer, cracked_imgs, crack_masks)
      train_losses.append(loss.item())

      ########### convert the logit output to probabilities using the sigmoid function!!!!!!!
      # convert probabilities to 0/1 with a threshold
      threshold = 0.5
      output_binary = (F.sigmoid(learnt_mask)>threshold).long()
      crack_masks = crack_masks.long()

      # calculate the training accuracies / mIoU
      train_bal_accs.append(balanced_accuracy(output_binary, crack_masks).item())
      train_mean_ious.append(mean_iou(output_binary, crack_masks, num_classes=2).mean().item())

      if i % 50 == 0:
        print(f"Epoch {epoch}: Avg Train Loss: {np.mean(train_losses):.4f}, Avg Train Bal Acc: {np.mean(train_bal_accs):.4f}, Avg Mean IoU: {np.mean(train_mean_ious):.4f}")
        path = os.path.join(working_dir, plot_dir)
        save_image(cracked_imgs[0], output_binary[0], crack_masks[0], epoch=epoch, step=i, plot_dir=path)

      i += 1

    # validation at the end of every epoch
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    val_m_ious = 0.0

    with torch.no_grad():
      for cracked_imgs, crack_masks in val_loader:
        cracked_imgs = cracked_imgs.to(device)
        crack_masks = crack_masks.to(device)

        learnt_mask = model(cracked_imgs)

        val_loss += segmentation_loss(learnt_mask, crack_masks).item()

        # convert logit output to binary
        output_binary = (F.sigmoid(learnt_mask)>0.5).long()
        crack_masks = crack_masks.long()

        # calcaulate the val accuracies / mIoU
        val_accuracy += balanced_accuracy(output_binary, crack_masks).item()
        val_m_ious += mean_iou(output_binary, crack_masks, num_classes=2).mean().item()

      val_loss /= len(val_loader)
      val_accuracy /= len(val_loader)
      val_m_ious /= len(val_loader)

      print(f"Epoch {epoch}: Avg Val Loss: {val_loss:.4f}, Avg balanced Accuracy: {val_accuracy:.4f}, Avg Mean IoU: {val_m_ious:.4f}")

      # save the training and val loss of this epoch
      val_loss_per_epoch.append(val_loss)
      train_loss_per_epoch.append(np.mean(train_losses))

      if val_loss < best_val_loss:
        best_val_loss = val_loss
        path = os.path.join(working_dir, f"models/{model_name}.pth")
        torch.save(model.state_dict(), path)
        print("Best model saved!")
        eps_without_improvement = 0
      else:
        eps_without_improvement += 1

      if eps_without_improvement > 10:
        break

      scheduler.step()

  return train_loss_per_epoch, val_loss_per_epoch