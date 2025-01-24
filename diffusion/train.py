import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from model import UNet
import os
from dataset import ImageDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Hyperparameters
IMAGE_FOLDER = "./new_dataset" 
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8  # 80% Train, 10% Validation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load full dataset
full_dataset = ImageDataset(IMAGE_FOLDER)

# Split into train and validation sets
train_size = int(TRAIN_SPLIT * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# Initialize model
model = UNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()  # Predicting noise, so MSE loss is used

T = 1000  # Number of diffusion steps
betas = torch.linspace(0.0001, 0.02, T).to(DEVICE)  # Linear schedule
alphas = 1 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)

# Function to sample noise from q(x_t | x_0)
def q_sample(x_0, t, noise):
    """
    Add noise to image x_0 at timestep t.
    Formula: q(x_t | x_0) = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
    """
    sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t])[:, None, None, None]
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t])[:, None, None, None]
    
    return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

# Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for x_0 in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True):
        x_0 = x_0.to(DEVICE)
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, T, (batch_size,), device=DEVICE)

        # Sample random noise
        noise = torch.randn_like(x_0)

        # Generate noisy image
        x_t = q_sample(x_0, t, noise)

        # Predict noise using the model
        pred_noise = model(x_t, t)

        # Compute loss
        loss = loss_fn(pred_noise, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

    # Validation Loop
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for x_0 in tqdm(val_loader, desc="Validating", leave=True):
            x_0 = x_0.to(DEVICE)
            batch_size = x_0.shape[0]

            # Sample random timesteps
            t = torch.randint(0, T, (batch_size,), device=DEVICE)

            # Sample noise and create noisy images
            noise = torch.randn_like(x_0)
            x_t = q_sample(x_0, t, noise)

            # Predict noise
            pred_noise = model(x_t, t)

            # Compute loss
            val_loss += loss_fn(pred_noise, noise).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

    # Save Model Checkpoint
    os.makedirs("checkpoints2", exist_ok=True)
    torch.save(model.state_dict(), f"checkpoints2/epoch_{epoch+1}.pth")
