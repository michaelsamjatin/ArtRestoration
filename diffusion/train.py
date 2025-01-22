import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from model import UNet, Diffusion
from torch.utils.data import random_split, DataLoader
from dataset_loader import get_dataset

def train(epochs=10, batch_size=1, learning_rate=1e-4, timesteps=500, img_dir="./paintings", mask_dir="./data/mask", plot_dir="./data/output", val_split=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load full dataset
    full_dataset = get_dataset(img_dir, mask_dir)  # Now returns a single dataset

    # Split into train and validation
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create dataloaders
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and diffusion process
    model = UNet().to(device)
    diffusion = Diffusion(timesteps=timesteps).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    os.makedirs(plot_dir, exist_ok=True)
    best_val_loss = float("inf")

    # Training loop
    for epoch in range(epochs):
        model.train()
        for i, batch in enumerate(dataloader_train):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            masks = (masks > 0.5).float()
            print(images.shape,masks.shape)
            t = torch.randint(0, timesteps, (images.shape[0],), device=device)

            noisy_images, noise = diffusion.add_noise(images, t, mask=masks)
            t = t.float() / timesteps
            t = t.view(-1, 1, 1, 1).expand(-1, 1, noisy_images.shape[2], noisy_images.shape[3])
            input_images = torch.cat([noisy_images, masks, t], dim=1)

            predicted_noise = model(input_images)
            loss = criterion(predicted_noise * masks, noise * masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader_train)}], Loss: {loss.item():.4f}")
                save_noise_images(noise, predicted_noise, epoch, i, plot_dir)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in dataloader_val:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                masks = (masks > 0.5).float()
                t = torch.randint(0, timesteps, (images.shape[0],), device=device)

                noisy_images, noise = diffusion.add_noise(images, t, mask=masks)
                t = t.float() / timesteps
                t = t.view(-1, 1, 1, 1).expand(-1, 1, noisy_images.shape[2], noisy_images.shape[3])
                input_images = torch.cat([noisy_images, masks, t], dim=1)

                predicted_noise = model(input_images)
                loss = criterion(predicted_noise * masks, noise * masks)
                val_loss += loss.item()

        val_loss /= len(dataloader_val)
        print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"./models/best_model.pth")
            print("Best model saved!")

        torch.save(model.state_dict(), f"./models/diffusion_inpainting_epoch_{epoch+1}.pth")
def save_noise_images(noise, predicted_noise, epoch, step, plot_dir):
    noise_image = noise[0].cpu().detach().numpy()  # Convert to numpy (shape: C x H x W)
    predicted_noise_image = predicted_noise[0].cpu().detach().numpy()

    noise_image = noise_image.transpose(1, 2, 0)
    predicted_noise_image = predicted_noise_image.transpose(1, 2, 0)

    # Create a directory for the current epoch if it doesn't exist
    os.makedirs(f"{plot_dir}/epoch_{epoch+1}", exist_ok=True)

    # Save the plots as images
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs[0].imshow(noise_image) 
    axs[0].set_title("Ground Truth Noise")
    axs[0].axis('off')  

    axs[1].imshow(predicted_noise_image) 
    axs[1].set_title("Predicted Noise")
    axs[1].axis('off')

    # Save the plot to the specified directory
    plt.savefig(f"{plot_dir}/epoch_{epoch+1}/step_{step}.png")
    plt.close()  
if __name__ == "__main__":
    train()