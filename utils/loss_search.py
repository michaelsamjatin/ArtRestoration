from multiprocessing import reduction
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import optuna
from torch.utils.data import DataLoader
from time import time
#from ptflops import get_model_complexity_info  # For FLOPs calculation

from models.unet import ShallowUNet
from tuning import get_data_loaders
from losses import DiceLoss, sigmoid_focal_loss


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loss Functions Setup
loss_functions = {
    "L1": nn.L1Loss(),
    "Focal": lambda x, m: sigmoid_focal_loss(x, m, reduction='mean'),
    "Dice": DiceLoss(),
    "L1_Focal": lambda x, y, m: nn.L1Loss()(x, y) + sigmoid_focal_loss(x, m, reduction='mean'),
    "L1_Dice": lambda x, y, m: nn.L1Loss()(x, y) + DiceLoss()(x, y, m),
    "Triple": lambda x, y, m: nn.L1Loss()(x, y) + sigmoid_focal_loss(x, m, reduction='mean') + DiceLoss()(x, y, m)
}

# Set fixed hyperparameters
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


# Optuna Objective Function (Updated)
def objective(trial):
    # Vary Loss Function
    loss_name = trial.suggest_categorical('loss_function', list(loss_functions.keys()))
    loss_fn = loss_functions[loss_name]

    # Init WandB
    wandb.init(
        project="crack-restoration",
        name="Loss_Experiment",
        config={
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "loss": loss_name
        }
    )

    # Setup Model, Optimizer, and Scheduler
    model = ShallowUNet()
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, verbose=True)

    # Data Loaders
    train_loader, val_loader = get_data_loaders(BATCH_SIZE)

    # Complexity Measurement: FLOPs and Parameters
    #macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True)
    #wandb.log({"FLOPs": macs, "Parameters": params})

    # Training Loop (Unchanged)
    for epoch in range(20):
        model.train()
        epoch_start = time()
        train_loss = 0.0

        for original, damaged, mask in train_loader:
            original, damaged, mask = damaged.to(device), mask.to(device), original.to(device)
            optimizer.zero_grad()

            # Forward Pass
            reconstructed = model(damaged)

            # Calculate Loss
            if loss_name in ['Dice', 'L1_Dice', 'L1_Focal', 'Triple']:
                # Supply mask
                loss = loss_fn(reconstructed, original, mask)
            else:
                # General Loss (L1 or L2)
                loss = loss_fn(reconstructed, original)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        epoch_time = time() - epoch_start

        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for original, damaged, mask in val_loader:
                original, damaged, mask = damaged.to(device), mask.to(device), original.to(device)
                reconstructed = model(damaged)

                # Calculate validation loss
                if loss_name in ['Dice', 'L1_Dice', 'Triple']:
                    # Supply mask
                    loss = loss_fn(reconstructed, original, mask)
                elif loss_name == 'Focal':
                    loss = loss_fn(reconstructed, mask)
                else:
                    # General Loss (L1 or L2)
                    loss = loss_fn(reconstructed, original)

                val_loss += loss.item()

                # Qualitative Logging
                if epoch % 10 == 0:
                    wandb.log({
                        "cracked_image": [wandb.Image(damaged[0].cpu(), caption="Cracked Input")],
                        "mask": [wandb.Image(mask[0].cpu(), caption="Mask")],
                        "ground_truth": [wandb.Image(original[0].cpu(), caption="Original")],
                        "prediction": [wandb.Image(reconstructed[0].cpu(), caption="Reconstruction")]
                    })

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Logging Performance and Complexity
        wandb.log({
            "Epoch": epoch,
            "Train Loss": train_loss,
            "Val Loss": val_loss,
            "Epoch Time": epoch_time,
        })

    # Finish WandB run
    wandb.finish()

    return val_loss



if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=6)

    # Print best trial
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
