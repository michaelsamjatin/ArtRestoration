import torch
import wandb
import tqdm
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from models.model_v2 import UNet
from utils.data import CrackDataset
from pytorch_ssim import ssim
from torchmetrics.functional import f1_score

def create_and_apply_mask(recon, clean, threshold=0.1):
    """
    Creates a mask with black background and white cracks
    """
    # Calculate absolute difference
    diff = torch.abs(recon - clean)
    
    # If image has multiple channels, take mean across channels
    if diff.dim() == 4:
        diff = diff.mean(dim=1)
    elif diff.dim() == 3:
        diff = diff.mean(dim=0)
    
    # Create black background
    result = torch.zeros_like(diff)
    
    # Set differences above threshold to white (1)
    result[diff > threshold] = 1
    
    # Scale to 0-255 range
    result = (result * 255).to(torch.uint8)
    
    return result


def train_step(model, optimizer, clean_image, cracked_image, crack_mask, device):
    optimizer.zero_grad()
    # Forward pass
    reconstruction = model(cracked_image)
    anomaly_mask = create_and_apply_mask(reconstruction.cpu(), clean_image.cpu())

    anomaly_mask = anomaly_mask.to(device)

    # Combination of SSIM (subtract from 1 to obtain proper loss metric) and MSE loss
    loss_ssim = (1 - ssim(reconstruction, clean_image))
    mse_loss = F.mse_loss(reconstruction, clean_image)
    # f1_loss = 0
    # f1_loss = f1_score(anomaly_mask / 255, crack_mask.mean(dim=1, keepdim=True).int(), task='binary') # identical dimensions for f1
    total_loss = loss_ssim + mse_loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    return total_loss, reconstruction, anomaly_mask, loss_ssim, mse_loss

def train_model(model, train_loader, val_loader, num_epochs, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        for clean_imgs, cracked_imgs, crack_masks in tqdm.tqdm(train_loader):

            clean_imgs = clean_imgs.to(device)
            cracked_imgs = cracked_imgs.to(device)
            crack_masks = crack_masks.to(device)
            
            # Training step
            loss, reconstruction, anomaly_mask, loss_ssim, mse_loss = train_step(model, optimizer, clean_imgs, cracked_imgs, crack_masks, device)
            train_losses.append(loss.item())


            # Log to wandb 
            wandb.log({
                'train_loss': loss.item(),
                'ssim_loss': loss_ssim.item(),
                'mse_loss': mse_loss.item(),
                # 'f1_loss': f1_loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            
            # Periodically log images
            if len(train_losses) % 100 == 0:
                wandb.log({
                    'images/clean': wandb.Image(clean_imgs[0].cpu()),
                    'images/cracked': wandb.Image(cracked_imgs[0].cpu()),
                    'images/reconstruction': wandb.Image(reconstruction[0].cpu()),
                    'images/crack_mask': wandb.Image(crack_masks[0].cpu()),
                    'images/anomly_mask': wandb.Image(anomaly_mask[0].cpu())
                })
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for clean_imgs, cracked_imgs, crack_masks in val_loader:
                clean_imgs = clean_imgs.to(device)
                cracked_imgs = cracked_imgs.to(device)
                
                reconstruction = model(cracked_imgs)
                val_loss = F.mse_loss(reconstruction, clean_imgs)
                val_losses.append(val_loss.item())
        
        avg_val_loss = np.mean(val_losses)
        # scheduler.step(avg_val_loss)

        # Log to wandb 
        wandb.log({
            'val_loss': val_loss.item()
        })
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, 'best_model.pth')
            
            # Log best model to wandb
            artifact = wandb.Artifact('best_model', type='model')
            artifact.add_file('best_model.pth')
            wandb.log_artifact(artifact)
        
        print(f"Epoch {epoch}: Avg Train Loss = {np.mean(train_losses):.4f}, Avg Val Loss = {avg_val_loss:.4f}")

if __name__ == '__main__':

    # Initialize wandb
    wandb.init(
        project="art-restoration",
        name="CrackDetection_with_attention_v4",
    )

    # Setup transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    # Create datasets
    train_dataset = CrackDataset(
        clean_dir='only_cracks/initial',
        cracked_dir='only_cracks/damaged_paintings',
        mask_dir='only_cracks/crack_mask',
        transform=transform
    )

    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = random_split(
    train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # For reproducibility
)

    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False, num_workers=4)

    # Initialize model and move to device
    # MPS for accelerated training on Apple Silicon, change to cuda if using HLR
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = UNet(in_channels=3).to(device)

    # Train model
    train_model(model, train_loader, val_loader, num_epochs=100, device=device)