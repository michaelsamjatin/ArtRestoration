import torch
import wandb
import tqdm
import numpy as np
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from model_v2 import UNet
from data import CrackDataset
from pytorch_ssim import ssim

def train_step(model, optimizer, clean_image, cracked_image, crack_mask):
    optimizer.zero_grad()
    # Forward pass
    reconstruction = model(cracked_image)

    # Combination of SSIM (subtract from 1 to obtain proper loss metric) and MSE loss
    reconstruction_loss = (1 - ssim(reconstruction, clean_image)) + F.mse_loss(reconstruction, clean_image)
    
    total_loss = reconstruction_loss
    
    # Backward pass
    total_loss.backward()
    optimizer.step()
    
    return total_loss, reconstruction

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
            loss, reconstruction = train_step(model, optimizer, clean_imgs, cracked_imgs, crack_masks)
            train_losses.append(loss.item())


            # Log to wandb 
            wandb.log({
                'train_loss': loss.item(),
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            
            # Periodically log images
            if len(train_losses) % 100 == 0:
                wandb.log({
                    'images/clean': wandb.Image(clean_imgs[0].cpu()),
                    'images/cracked': wandb.Image(cracked_imgs[0].cpu()),
                    'images/reconstruction': wandb.Image(reconstruction[0].cpu()),
                    'images/crack_mask': wandb.Image(crack_masks[0].cpu())
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
        name="CrackDetection_with_attention_v3",
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

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(train_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Initialize model and move to device
    # MPS for accelerated training on Apple Silicon, change to cuda if using HLR
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = UNet(in_channels=3).to(device)

    # Train model
    train_model(model, train_loader, val_loader, num_epochs=100, device=device)