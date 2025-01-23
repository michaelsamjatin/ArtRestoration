import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure


class L1_SSIM_Loss(nn.Module):
    """
    Combines L1 Loss and SSIM Loss.

    Parameters:
        - alpha (float): Weighting factor to balance L1 and SSIM losses.
    """
    def __init__(self, alpha=0.7):
      super(L1_SSIM_Loss, self).__init__()

      # Init parameter
      self.alpha = alpha

      # Init losses
      self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
      self.l1 = nn.L1Loss()

    def forward(self, reconstructed, original):
      """
      Compute loss for predicted reconstruction and original image.
      """
      # L1 Loss
      l1_loss = self.l1(reconstructed, original)

      # SSIM Loss
      ssim_value = self.ssim(reconstructed, original)
      ssim_loss = 1 - ssim_value

      # Combine the losses with respective weights
      total_loss = self.alpha * l1_loss + (1 - self.alpha) * ssim_loss

      return total_loss