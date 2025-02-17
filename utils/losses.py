import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchvision.ops import sigmoid_focal_loss


class L1_SSIM_Loss(nn.Module):
    """
    Combines L1 Loss and SSIM Loss.

    Parameters:
        - alpha (float): Weighting factor to balance L1 and SSIM losses.
    """
    def __init__(self, alpha=0.7, beta=0.7):
      super(L1_SSIM_Loss, self).__init__()

      # Init parameter
      self.alpha = alpha
      self.beta = beta

      # Init losses
      self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to('cuda')
      self.l1 = nn.L1Loss()


    def forward(self, reconstructed, original, mask=None):
        """
        Compute loss for predicted reconstruction and original image.
        """
        # L1 Loss
        l1_loss = self.l1(reconstructed, original)

        # SSIM Loss
        ssim_value = self.ssim(reconstructed, original)
        ssim_loss = 1 - ssim_value

        # Apply mask
        if mask is not None:
            recon_masked = reconstructed * mask
            original_masked = reconstructed * mask
        
            # Compute l1 loss between masked areas
            l1_masked = self.l1(recon_masked, original_masked)
            l1_loss = (1 - self.beta) * l1_masked + self.beta * l1_loss
        
        # Combine the losses with respective weights
        total_loss = self.alpha * l1_loss + (1 - self.alpha) * ssim_loss

        return total_loss



class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, restored, original, mask):
        restored_masked = restored * mask
        original_masked = original * mask
        
        restored_masked = torch.sigmoid(restored_masked)
        restored_masked = restored_masked.view(-1)
        original_masked = original_masked.view(-1)
        
        intersection = (restored_masked * original_masked).sum()
        dice = (2. * intersection + self.smooth) / (restored_masked.sum() + original_masked.sum() + self.smooth)
        
        return 1 - dice





class TripletLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', smooth=1e-6, device='cuda'):
        super().__init__()

        # Init losses
        self.l1 = nn.MAELoss().to(device)
        self.focal = lambda x, y: sigmoid_focal_loss(x, y, alpha, gamma, reduction).to(device)
        self.dice = DiceLoss(smooth).to(device)


    def forward(self, restored, original, mask):
        # Compute individual losses
        l1_score = self.l1(restored, original)
        focal_score = self.focal(restored, original)
        dice_score = self.dice(restored, original, mask)

        # Aggregate
        total_loss = l1_score + focal_score + dice_score

        return total_loss.item()