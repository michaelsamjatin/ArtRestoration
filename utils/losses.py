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
    def __init__(self, alpha=0.7, beta=4.0):
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
            l1_loss = self.beta * l1_masked + 0.8 * l1_loss
        
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







### Masked Loss Functions ###

def masked_l1(pred, target, mask, weight=4.0):
    diff = torch.abs(pred - target)
    weighted_diff = torch.where(mask > 0.5, diff * weight, diff * 0.8)
    return torch.sum(weighted_diff) / (torch.sum(mask)*weight + torch.sum(1-mask)*0.8)

"""
def masked_ssim(pred, target, mask, window_size=7):
    mask = mask.repeat(1,3,1,1)  # RGB channels
    ux = _gaussian_filter(pred*mask, window_size) 
    uy = _gaussian_filter(target*mask, window_size)
    ...  # Full masked SSIM computation
    return (ssim_map * mask).sum() / mask.sum()

def connectivity_ssim(pred, target, mask):
    base_ssim = masked_ssim(pred, target, mask)
    
    # Sobel gradients
    grad_x = F.conv2d(pred, sobel_x) 
    grad_y = F.conv2d(pred, sobel_y)
    pred_grad = torch.sqrt(grad_x**2 + grad_y**2)
    
    target_grad = F.conv2d(target, sobel_x)**2 + F.conv2d(target, sobel_y)**2
    grad_loss = F.l1_loss(pred_grad, target_grad)
    
    return 0.7*base_ssim + 0.3*(1 - grad_loss)


class CrackLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2):
        super().__init__()
        self.alpha = alpha  # mL1 weight
        self.beta = beta    # mSSIM weight
        self.gamma = gamma  # Connectivity weight
        
    def forward(self, pred, target, mask):
        l1 = masked_l1(pred, target, mask)
        ssim = 1 - masked_ssim(pred, target, mask)
        connectivity = 1 - connectivity_ssim(pred, target, mask)
        return self.alpha*l1 + self.beta*ssim + self.gamma*connectivity
"""


class MultiScaleLoss(nn.Module):
    def __init__(self, scales=[1, 0.5, 0.25], weights=[0.6, 0.3, 0.1]):
        super().__init__()
        self.scales = scales
        self.weights = weights
    
    def forward(self, pred, target, mask):
        loss = 0
        for scale, weight in zip(self.scales, self.weights):
            p = F.interpolate(pred, scale_factor=scale)
            t = F.interpolate(target, scale_factor=scale)
            m = F.interpolate(mask, scale_factor=scale)
            loss += weight * masked_l1(p, t, m)
        return loss
