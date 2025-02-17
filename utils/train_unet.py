import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
import time
from pathlib import Path
import wandb

from models.unet import AttentionUNet
import utils.losses as losses
from utils.data import get_dataset


class Solver():

    def __init__(self, model, data, **kwargs):
        """
        Creates a solver for classification.

        Parameters:
            - model (nn.Module):
                  Model to be trained.
            - data (dict):
                  Training and validation datasets.
                  Dictionary with keys `train` for training set and `val` for validation set.
            - loss (CustomLoss):
                  Custom Loss Function
                  [Default: None]
            - loss_config (dict|None):
                  Dictionary with keyword arguments for calling the loss function.
                  [Default: {}]
            - optimizer (str):
                  Class name of the optimizer to be used.
                  [Default: 'AdamW']
            - optimizer_config (dict):
                  Dictionary with keyword arguments for calling for the optimizer.
                  Model parameters don't have to be passed explicitly.
                  [Default: {'lr': 1e-3}]
            - batch_size (int):
                  Number of samples per minibatch.
                  [Default: 16]
            - num_train_samples (int):
                  Number of training samples to be used for evaluation.
                  [Default: 1000]
            - num_val_samples (int|None):
                  Number of validation samples to be used for evaluation.
                  If parameter is `None`, all samples in the given validation set are used.
                  [Default: None]
            - scheduler (str|None):
                  Class name of the learning rate scheduler to be used.
                  If parameter is not given or `None`, no scheduler is used.
                  [Default: None]
            - scheduler_config (dict):
                  Dictionary with keyword arguments to provide for the scheduler.
                  The optimizer is passed in automatically.
                  [Default: {}]
            - metric (str):
                  Metric to be used for measure performance. Torchmetrics class.
                  [Default: 'image.StructuralSimilarityIndexMeasure']
            - metric_config (dict):
                  Dictionary with keyword arguments for calling the metric.
                  [Default: {'data_range': 1.0}]
            - model_name (str):
                  Name to be used for saving the model. It will also be used for the JobID.
                  [Default: Model Class Name]
        """
        
        self.model = model

        # Store training and validation data
        self.data_train = data['train']
        self.data_val = data['val']

        # Set device.
        device_name = kwargs.pop('device')
        self.device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

        # Root Path.
        self.log_root = kwargs.pop('log_root')

        # Workers.
        self.workers = kwargs.pop('num_workers')

        # Model Name.
        self.model_name = kwargs.pop('model_name')

        # Loss function and configuration.
        loss = getattr(losses, kwargs.pop('loss'))
        loss_config = kwargs.pop('loss_config')
        self.loss = loss(**loss_config).to(self.device)
        

        # Optimizer.
        optimizer = getattr(torch.optim, kwargs.pop('optimizer'))
        self.optimizer = optimizer(model.parameters(), **kwargs.pop('optimizer_config'))

        # Scheduler. (optional)
        self.scheduler = kwargs.pop('scheduler')
        if self.scheduler:
            scheduler = getattr(torch.optim.lr_scheduler, self.scheduler)
            self.scheduler = scheduler(self.optimizer, **kwargs.pop('scheduler_config'))

        # Metric.
        metric_name = kwargs.pop('metric')
        metric_config = kwargs.pop('metric_config', {})
        self.init_metric(metric_name, metric_config)

        # Store remaining arguments.
        self.__dict__ |= kwargs

        # Bookkeeping for training state.
        self.epoch = 0
        self.num_epochs = 0
        self.loss_history = []
        self.train_score = []
        self.val_score = []
        self.lr_history = [] if self.scheduler else None

    def init_metric(self, metric_name, metric_config):
        """
        Helper function to initialize metrics from different paths within torchmetrics
        """
        try:
            if '.' in metric_name:
                module_name, metric_class = metric_name.rsplit('.', 1)
                metric_module = getattr(torchmetrics, module_name)
                metric = getattr(metric_module, metric_class)
            else:
                metric = getattr(torchmetrics, metric_name)
        
            # Initialize the metric
            self.metric = metric(**metric_config).to(self.device)
            
        except (ImportError, AttributeError) as e:
            raise ValueError(f"Unable to initialize metric '{metric_name}'. Ensure it's a valid metric.") from e


    def step(self, damaged, mask, original):
        """
        Performs a single training step.

        Parameters:
            - damaged (Tensor): Damaged input image.
            - mask (Tensor): Mask used to create damage.
            - original (Tensor): Original image.

        Returns:
            - reconstruction (Tensor): Predicted reconstruction.
            - loss (float): Loss value.
        """
        # Reset gradients
        self.optimizer.zero_grad()

        # Compute forward pass
        reconstruction = self.model(damaged)
    
        # Compute loss
        loss = self.loss(reconstruction, original)
    
        # Compute backward pass
        loss.backward()
        self.optimizer.step()

        return reconstruction, loss.item()


    def save(self, path):
        """
        Save model and training state to disk.

        Parameters:
            - path (str): Path to store checkpoint.

        """
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'num_epochs': self.num_epochs,
            'loss_history': self.loss_history,
            'train_score': self.train_score,
            'val_score': self.val_score,
            'lr_history': self.lr_history
        }

        # Save learning rate scheduler state if defined.
        if self.scheduler:
            checkpoint['scheduler'] = self.scheduler.state_dict()

        # Save checkpoint to disk.
        torch.save(checkpoint, path)


    def load(self, path):
        """
        Load checkpoint from disk.

        Parameters:
            - path (str): Path to checkpoint.

        """
        checkpoint = torch.load(path, map_location=torch.device(self.device))

        # Load model and optimizer state.
        self.model.load_state_dict(checkpoint.pop('model'))
        self.optimizer.load_state_dict(checkpoint.pop('optimizer'))

        # Load learning rate scheduler state if defined.
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint.pop('scheduler'))

        # Load the remaining attributes.
        self.__dict__ |= checkpoint
        

    def test(self, data_test, with_loss=False):
        """
        Compute the performance of the model based on provided metric.

        Parameters:
            - dataset (torch.Tensor): Dataset for testing.

        Returns:
            - metric score (float): Score based on provided metric.

        """
        # Prepare for testing
        self.model.to(self.device)
        self.model.eval()
        self.metric.reset()

        # Create loader for dataset.
        data_loader = DataLoader(
            data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers
        )

        # Init list for validation loss
        val_losses = []

        with torch.no_grad():
            for original, damaged, mask in data_loader:
                # Transfer data to selected device.
                damaged = damaged.to(self.device)
                original = original.to(self.device)
                mask = mask.to(self.device)

                # Compute forward pass.
                reconstructed = self.model(damaged)

                self.metric(reconstructed, original)

                # Compute loss
                if with_loss:
                    val_losses.append(self.loss(reconstructed, original).item())

        # Qualitative Logging
        if self.epoch % 10 == 0:
            wandb.log({
                "cracked_image": [wandb.Image(damaged[0].cpu(), caption="Cracked Input")],
                "mask": [wandb.Image(mask[0].cpu(), caption="Mask")],
                "ground_truth": [wandb.Image(original[0].cpu(), caption="Original")],
                "prediction": [wandb.Image(reconstructed[0].cpu(), caption="Reconstruction")]
            })
        
        # Accumulate metric.
        result = self.metric.compute()

        # Reset metric.
        self.metric.reset()

        # Reset model.
        self.model.train()

        if with_loss:
            return result.item(), np.mean(val_losses)
        else: 
            return result.item()


    def train(self, num_epochs, verbose=False):
        """
        Train the model for given number of epochs.

        Parameters:
            - num_epochs (int): Number of epochs to train.

        Returns:
            - history (dict):
                - loss: Training set loss per epoch.
                - train_score: Training set accuracy per epoch.
                - val_score: Validation set accuracy per epoch.
                - lr (optional): Learning Rate per epoch if using a scheduler.
        """
        self.model.to(self.device)
        self.model.train()
        self.num_epochs += num_epochs

        # Data loader for training set.
        train_loader = DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers
        )

        # Init best validation score and best params
        best_val_loss = 0
        best_params = None

        weights_root = Path(self.log_root / 'weights/unet/')

        # Start Training
        for epoch in tqdm(range(num_epochs)):
            self.epoch += 1
            loss_history = []
            epoch_start = time.time()
            
            for i, (original, damaged, mask) in enumerate(train_loader):
                
                # Transfer inputs and labels to selected device.
                damaged = damaged.to(self.device)
                original = original.to(self.device)
                mask = mask.to(self.device)

                # Training Step
                reconstruction, loss = self.step(damaged, mask, original)

                # Store current loss.
                loss_history.append(loss)

            # Store epoch time
            epoch_time = time.time() - epoch_start

            # Store average loss per epoch.
            train_loss = sum(loss_history) / i
            self.loss_history.append(train_loss)

            # Store training accuracy.
            train_score = self.test(self.data_train)
            self.train_score.append(train_score)

            # Store validation accuracy.
            val_score, val_loss = self.test(self.data_val, with_loss=True)
            self.val_score.append(val_score)

            # Update learning rate.
            if self.scheduler:
                self.scheduler.step()
                lr = self.scheduler.get_last_lr()[0]

            # Update best accuracy and model parameters.
            if val_loss > best_val_loss:
                # Update best validation score and model
                best_val_loss = val_loss
                best_model = f"{self.model_name}_{self.epoch}epochs.pth"
                
                # Save on server
                self.save(weights_root / best_model)

            # Periodically save checkpoints
            elif epoch >= 50 and epoch % 10 == 0: 
                # Save on server
                self.save(weights_root / f"{self.model_name}_{self.epoch}epochs.pth")

            # Log to wandb
            wandb.log({
                'train_loss': train_loss,
                'train_score': train_score,
                'val_score': val_score,
                'val_loss': val_loss,
                'lr': lr,
                'epoch_time': epoch_time
            })

        # Save final model
        final_model = f"{self.model_name}_{self.epoch}epochs.pth"
        self.save(weights_root / final_model)

        # Log final model to wandb
        artifact = wandb.Artifact('final_model', type='model')
        artifact.add_file(weights_root / final_model)
        wandb.log_artifact(artifact)

        # Log best model to wandb
        artifact = wandb.Artifact('best_model', type='model')
        artifact.add_file(weights_root / best_model)
        wandb.log_artifact(artifact)

        # Swap best parameters from training into the model.
        self.model.load_state_dict(best_params)



def run_training(config):

    # Initialize wandb
    wandb.init(
        project="crack-restoration",
        name="FinalUNet",
        config=config,
        dir=config['log_root']
    )  

    # Setup transformation ### 
    transform = transforms.Compose([
        #transforms.Resize((512, 512)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # Fetch dataset
    data_train, data_val, data_test = get_dataset( 
        Path(config['data_root']), 
        transform, 
        train_ratio=0.7, 
        val_ratio=0.1, 
        test_ratio=0.2
    )

    # Prepare data for training
    data = {
        'train': data_train,
        'val': data_val
    }

    # Init generator
    generator = AttentionUNet(input_channels=3)

    # Init GAN solver
    solver = Solver(
        generator,
        data,
        model_name=config['model_name'],
        device=config['device'],
        log_root=Path(config['log_root']),
        num_workers=config['num_workers'],
        batch_size=config['batch_size'],
        metric=config['metric'],
        metric_config=config['metric_config'],
        optimizer=config['optimizer'],
        optimizer_config=config['optimizer_config'],
        scheduler=config['scheduler'],
        scheduler_config=config['scheduler_config'],
        loss=config['loss'],
        loss_config=config['loss_config'],
    )


    # Training
    solver.train(config['epochs'])

    # Testing
    test_score = solver.test(data_test)

    # Log test score to wandb
    wandb.log({
        'test_score': test_score
    })

    wandb.finish()


def run_testing(config):
    # Setup transformation 
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # Fetch dataset
    data_train, data_val, data_test = get_dataset( 
        Path(config['data_root']), 
        transform, 
        train_ratio=0.7, 
        val_ratio=0.1, 
        test_ratio=0.2
    )

    # Prepare data for training
    data = {
        'train': data_train,
        'val': data_val
    }

    # Init generator
    generator = AttentionUNet(input_channels=3)

    # Init GAN solver
    solver = Solver(
        generator,
        data,
        model_name=config['model_name'],
        device=config['device'],
        log_root=Path(config['log_root']),
        num_workers=config['num_workers'],
        batch_size=config['batch_size'],
        metric=config['metric'],
        metric_config=config['metric_config'],
        optimizer=config['optimizer'],
        optimizer_config=config['optimizer_config'],
        scheduler=config['scheduler'],
        scheduler_config=config['scheduler_config'],
        loss=config['loss'],
        loss_config=config['loss_config'],
    )


    # Load weights
    model_path = Path("/home/artproject/weights/encdec/___.pth")
    solver.load(model_path)

    # Testing
    solver.evaluation(data_test, save_dir=Path("/home/artproject/results"))




if __name__ == "__main__":
    # Define config
    config = {
        "model_name": 'AttentionUNet',
        "device": "cuda",
        "batch_size": 16,
        "num_workers": 2,
        "epochs": 100,
        "optimizer": 'AdamW',
        "optimizer_config": {'lr': 1e-3, 'weight_decay': 3e-4},
        "loss": 'L1_SSIM_Loss',
        "loss_config": {'alpha': 0.7},
        "scheduler": 'CosineAnnealingLR',
        "scheduler_config": {'T_max': 100, 'eta_min':1e-5},
        "metric": 'image.StructuralSimilarityIndexMeasure',
        "metric_config": {'data_range': 1.0},
        "data_root": "/home/artproject/data",
        "log_root": "/home/artproject",
    }

    run_training(config)

    # run_testing(config)