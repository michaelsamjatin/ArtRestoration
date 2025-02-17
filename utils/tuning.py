import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
import optuna
from pathlib import Path

from utils.data import get_dataset 
from models.unet import ShallowUNet

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup DataLoaders
def get_data_loaders(batch_size):
    # Setup transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # Fetch dataset
    data_train, data_val, data_test = get_dataset( 
        Path("/home/artproject/data"), 
        transform, 
        train_ratio=0.7, 
        val_ratio=0.1, 
        test_ratio=0.2
    )
    
    # DataLoaders
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# Training and Validation Loop
def train_and_validate(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for damaged, original in train_loader:
            damaged, original = damaged.to(device), original.to(device)
            optimizer.zero_grad()
            reconstructed = model(damaged)
            loss = criterion(reconstructed, original)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Log training loss
        train_loss = running_loss / len(train_loader)
        wandb.log({"train_loss": train_loss, "epoch": epoch})

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for damaged, original in val_loader:
                damaged, original = damaged.to(device), original.to(device)
                reconstructed = model(damaged)
                loss = criterion(reconstructed, original)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Log validation metrics
        wandb.log({
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })

    return val_loss

# Optuna Objective Function
def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])

    # Data Loaders
    train_loader, val_loader = get_data_loaders(batch_size)

    # Model, Loss, Optimizer, and Scheduler
    model = ShallowUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # Init WandB
    wandb.init(
        project="crack-restoration",
        config={
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau"
        }
    )
    
    # Training and Validation
    val_loss = train_and_validate(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs=10)
    
    # Finish WandB run
    wandb.finish()
    
    return val_loss



if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    # Print best trial
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
