
import os 
from pathlib import Path
import json

import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchmetrics
from torchvision.utils import save_image

from tqdm import tqdm
import wandb

from models.gan import UNetGenerator, Discriminator, PatchGANDiscriminator
import utils.losses as losses
from utils.data import get_dataset_gan

from PIL import Image
import matplotlib.pyplot as plt


class Solver():

    def __init__(self, generator, data, **kwargs):

        # Store training and validation data
        self.data_train = data['train']
        self.data_val = data['val']

        # Set device.
        device_name = kwargs.pop('device')
        self.device = torch.device(device_name if torch.cuda.is_available() else 'cpu')

        # Name
        self.model_name = kwargs.pop('model_name') 
       
        # Log Root
        self.log_root = kwargs.pop('log_root')

        # Workers.
        self.workers = kwargs.pop('num_workers')

        # Batch Size
        self.batch_size = kwargs.pop('batch_size')

        # Loss weights
        self.loss_weights = kwargs.pop('loss_weights')

        # Step for losses
        self.step = kwargs.pop('loss_step', None) 

        # Metric.
        metric_name = kwargs.pop('metric')
        metric_config = kwargs.pop('metric_config', {})
        self.init_metric(metric_name, metric_config)

        # Loss function and configuration.
        self.adversarial_loss = self.init_loss(kwargs.pop('adversarial_loss'), kwargs.pop('adversarial_config')).to(self.device)
        self.reconstruction_loss = self.init_loss(kwargs.pop('reconstruction_loss'), kwargs.pop('reconstruction_config')).to(self.device)

        # Init models
        self.generator = generator.to(self.device)
        self.global_discriminator = Discriminator().to(self.device)
        self.local_discriminator = PatchGANDiscriminator().to(self.device)

        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.global_discriminator.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.local_discriminator.parameters(), max_norm=1.0)

        self.generator.apply(self.weights_init)
        self.global_discriminator.apply(self.weights_init)
        self.local_discriminator.apply(self.weights_init)


        # Optimizer.
        optimizer = getattr(torch.optim, kwargs.pop('optimizer'))
        self.optimizer_G = optimizer(self.generator.parameters(), **kwargs.pop('optimizer_config_gan'))
        self.optimizer_D_global = optimizer(self.global_discriminator.parameters(), **kwargs.pop('optimizer_config_dg'))
        self.optimizer_D_local = optimizer(self.local_discriminator.parameters(), **kwargs.pop('optimizer_config_dl'))

        # Store remaining arguments.
        self.__dict__ |= kwargs

        # Bookkeeping for training state.
        self.epoch = 0
        self.num_epochs = 0
        self.train_score = []
        self.val_score = []


    def init_loss(self, loss_name: str, loss_config: dict):
        """
        Initializes loss function. It will first look in the losses module containing
        custom loss functions and then in the torch.nn library.
        """
        try:
            loss = getattr(losses, loss_name)
        except AttributeError:
            try:
                loss = getattr(torch.nn, loss_name)
            except AttributeError:
                raise AttributeError(f"Loss function '{loss_name}' not found in 'losses' or 'torch.nn' modules.")
        
        return loss(**loss_config)


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

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)


    def save(self, path):
        """
        Save model and training state to disk.

        Parameters:
            - path (str): Path to store checkpoint.

        """

        # Create checkpoint
        checkpoint = {
            'generator': self.generator.state_dict(),
            'gobal_discriminator': self.global_discriminator.state_dict(),
            'local_discriminator': self.local_discriminator.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D_global': self.optimizer_D_global.state_dict(),
            'optimizer_D_local': self.optimizer_D_local.state_dict(),
            'epoch': self.epoch,
            'num_epochs': self.num_epochs,
            #'loss_history': self.loss_history,
            'train_score': self.train_score,
            'val_score': self.val_score
        }

        # Save checkpoint to disk.
        torch.save(checkpoint, path)


    def load(self, path):
        """
        Load checkpoint from disk.

        Parameters:
            - path (str): Path to checkpoint.

        """

        # Load checkpoint.
        checkpoint = torch.load(path, map_location=torch.device(self.device))

        # Load model states.
        self.generator.load_state_dict(checkpoint.pop('generator'))
        self.global_discriminator.load_state_dict(checkpoint.pop('gobal_discriminator'))
        self.local_discriminator.load_state_dict(checkpoint.pop('local_discriminator'))

        # Load optimizer state.
        self.optimizer_G.load_state_dict(checkpoint.pop('optimizer_G'))
        self.optimizer_D_global.load_state_dict(checkpoint.pop('optimizer_D_global'))
        self.optimizer_D_local.load_state_dict(checkpoint.pop('optimizer_D_local'))

        # Load the remaining attributes.
        self.__dict__ |= checkpoint


    def train_step_generator(self, reconstruction, original, tag='total'):
        """Perform training step for generator."""

        self.optimizer_G.zero_grad()
    
        # Adversarial loss
        output_global = self.global_discriminator(reconstruction)
        global_loss = self.adversarial_loss(
            output_global, torch.ones_like(output_global)
        ) if tag in ['discriminator', 'global', 'total'] else 0

        output_local = self.local_discriminator(reconstruction)
        local_loss = self.adversarial_loss(
            output_local, torch.ones_like(output_local)
        ) if tag in ['discriminator', 'local', 'total'] else 0
        
        # Reconstruction loss 
        recon_loss = self.reconstruction_loss(
            reconstruction, original
        ) if tag in ['recon', 'total'] else 0
        
        # Extract loss weights
        global_weight = self.loss_weights['global']
        local_weight = self.loss_weights['local']
        recon_weight = self.loss_weights['recon']

        # Combine losses
        g_loss = global_weight * global_loss + local_weight * local_loss + recon_weight * recon_loss
        
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss


    def train_step_discriminator(self, optimizer, discriminator, reconstruction, original):
        """Perform training step for discriminator."""

        optimizer.zero_grad()
    
        # Real images
        output_original = discriminator(original)
        real_loss = self.adversarial_loss(
            output_original, torch.full_like(output_original, 0.9)  # simple one-sided label smoothing
        )
        
        # Fake images (detach generator gradients)
        fake_images = reconstruction.detach()
        output_fake = discriminator(fake_images)
        fake_loss = self.adversarial_loss(
            output_fake, torch.zeros_like(output_fake)
        )
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer.step()
        
        return d_loss

    def generate_image(self, original, damaged, restoration, path): 
    
        # Convert tensors to numpy arrays for visualization
        ground_truth_img = original[0].cpu().numpy().transpose(1, 2, 0)
        damaged_img = damaged[0].cpu().numpy().transpose(1, 2, 0)
        restored_img = restoration[0].cpu().numpy().transpose(1, 2, 0)
    
        # Clip and normalize images to [0, 1] range
        damaged_img = np.clip((damaged_img + 1) / 2, 0, 1)  # Assuming images are normalized to [-1, 1]
        ground_truth_img = np.clip((ground_truth_img + 1) / 2, 0, 1)
        restored_img = np.clip((restored_img + 1) / 2, 0, 1)
    
        # Upscale the restored image to the original size (e.g., 512x512)
        damaged_img_upscaled = np.array(Image.fromarray((damaged_img * 255).astype(np.uint8)).resize((512, 512), Image.BILINEAR)) / 255
        ground_truth_img_upscaled = np.array(Image.fromarray((ground_truth_img * 255).astype(np.uint8)).resize((512, 512), Image.BILINEAR)) / 255
        restored_img_upscaled = np.array(Image.fromarray((restored_img * 255).astype(np.uint8)).resize((512, 512), Image.BILINEAR)) / 255
    
    
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        images = [ground_truth_img_upscaled, damaged_img_upscaled, restored_img_upscaled]
        titles = ["Original", "Damaged", "Reconstruction"]
    
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img, cmap='gray')  # Remove cmap='gray' if the images are in color.
            ax.set_title(title)
            ax.axis('off')  # Hide axes for a cleaner look
    
        plt.tight_layout()  # Adjust layout to prevent overlap
    
        # Save the figure to a desired path.
        plt.savefig(path, dpi=300)
        plt.close(fig)


    def evaluation(self, data_test, save_dir="results"):
        """Method to save model reconstructions on test data for qualitative analysis"""

        # Prepare for testing
        self.generator.to(self.device)
        self.generator.eval()

        # Create base directories for saving images
        model_name = f"{self.model_name}_{self.epoch}epochs"
        os.makedirs(f"{save_dir}/{model_name}/original", exist_ok=True)
        os.makedirs(f"{save_dir}/{model_name}/damaged", exist_ok=True)
        os.makedirs(f"{save_dir}/{model_name}/reconstructed", exist_ok=True)

        # Create loader for dataset.
        data_loader = DataLoader(
            data_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers
        )

        # Global index for consistent naming
        image_index = 0 

        with torch.no_grad():
            for damaged, original in data_loader:
                # Move data to device
                damaged = damaged.to(self.device)
                original = original.to(self.device)

                # Compute forward pass
                reconstruction = self.generator(damaged)

                # Save images
                for i in range(damaged.size(0)):
                    save_image(original[i], f"{save_dir}/{model_name}/original/{image_index}.jpg")
                    save_image(damaged[i], f"{save_dir}/{model_name}/damaged/{image_index}.jpg")
                    save_image(reconstruction[i], f"{save_dir}/{model_name}/reconstructed/{image_index}.jpg")
                    image_index += 1

        # Reset model
        self.generator.train()


    def test(self, data_test, with_loss=False, print_sample=False):
        """
        Compute the performance of the generator based on provided metric.

        Parameters:
            - dataset (torch.Tensor): Dataset for testing.

        Returns:
            - metric score (float): Score based on provided metric.

        """
        # Prepare for testing
        self.generator.to(self.device)
        self.generator.eval()
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
            for damaged, original in data_loader:
                # Move data to device
                damaged = damaged.to(self.device)
                original = original.to(self.device)

                # Compute forward pass.
                reconstruction = self.generator(damaged)

                self.metric(reconstruction, original)

                if print_sample: 
                    self.generate_image(original, damaged, reconstruction, self.log_root / f"images/{self.model_name}_{self.epoch}.png")
                    print_sample = False
                
                # Compute loss
                if with_loss:
                    val_losses.append(self.reconstruction_loss(reconstruction, original).item())

        # Accumulate metric.
        result = self.metric.compute()

        # Reset metric.
        self.metric.reset()

        # Reset model.
        self.generator.train()

        if with_loss:
            return result.item(), np.mean(val_losses)
        else: 
            return result.item()


    def train(self, num_epochs):
        """
        Train the model for given number of epochs.

        Parameters:
            - num_epochs (int): Number of epochs to train.

        Returns:
            - history (dict):
                - d_loss_global: Training set loss per epoch.
                - train_score: Training set accuracy per epoch.
                - val_score: Validation set accuracy per epoch.
                - lr (optional): Learning Rate per epoch if using a scheduler.
        """

        # Data loader for training set.
        train_loader = DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers
        )

        # Keep track of best model
        best_val_loss = float('inf')
        best_model = ''

        # Init weight root
        weights_root = Path(self.log_root / 'weights/gan/')

        # Training Loop
        for epoch in tqdm(range(num_epochs)):
            self.epoch += 1

            for i, (damaged, original) in enumerate(train_loader):

                # Move data to device
                damaged = damaged.to(self.device)
                original = original.to(self.device)

                # Predict reconstruction
                reconstruction = self.generator(damaged)

                # Step
                d_loss_global = self.train_step_discriminator(self.optimizer_D_global, self.global_discriminator, reconstruction, original)
                d_loss_local = self.train_step_discriminator(self.optimizer_D_local, self.local_discriminator, reconstruction, original)

                if self.step and (i % self.step * 3) == 0:
                    tag = 'recon'
                elif self.step and (i % self.step * 3) == 1:
                    tag = 'discriminator'
                else:
                    tag = 'total'

                g_loss = self.train_step_generator(reconstruction, original, tag=tag)


            # Store training accuracy.
            train_score = self.test(self.data_train)
            self.train_score.append(train_score)

            # Store validation accuracy.
            val_score, val_loss = self.test(self.data_val, with_loss=True, print_sample=(epoch % 10 == 0))
            self.val_score.append(val_score)

            # Log to wandb 
            wandb.log({
                'd_loss_global': d_loss_global.item(),
                'd_loss_local': d_loss_local.mean().item(),
                'g_loss': g_loss.item(),
                'train_score': train_score,
                'val_score': val_score,
                'val_loss': val_loss
            })

            # Save checkpoint for best model
            if val_loss < best_val_loss:
                # Update best validation score and model
                best_val_loss = val_loss
                best_model = f"{self.model_name}_{self.epoch}epochs.pth"
                
                # Save on server
                self.save(weights_root / best_model)

            # Periodically save checkpoints
            elif epoch >= 50 and epoch % 10 == 0: 
                # Save on server
                self.save(weights_root / f"{self.model_name}_{self.epoch}epochs.pth")


        # Log best model to wandb
        artifact = wandb.Artifact('best_model', type='model')
        artifact.add_file(weights_root / best_model)
        wandb.log_artifact(artifact)



def run_training(config):

    # Initialize wandb
    wandb.init(
        project="art-restoration",
        name=config['model_name'],
        config=config,
        dir=config['log_root']
    )  

    # Setup transformation ### 
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

    # Fetch dataset
    data_train, data_val, data_test = get_dataset_gan( 
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
    generator = UNetGenerator()

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
        optimizer_config_gan=config['optimizer_config_gan'],
        optimizer_config_dg=config['optimizer_config_dg'],
        optimizer_config_dl=config['optimizer_config_dl'],
        adversarial_loss=config['adversarial_loss'],
        adversarial_config=config['adversarial_config'],
        reconstruction_loss=config['reconstruction_loss'],
        reconstruction_config=config['reconstruction_config'],
        loss_weights=config['loss_weights'],
        loss_step=config['loss_step']
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
    data_train, data_val, data_test = get_dataset_gan( 
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
    generator = UNetGenerator()

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
        optimizer_config_gan=config['optimizer_config_gan'],
        optimizer_config_dg=config['optimizer_config_dg'],
        optimizer_config_dl=config['optimizer_config_dl'],
        adversarial_loss=config['adversarial_loss'],
        adversarial_config=config['adversarial_config'],
        reconstruction_loss=config['reconstruction_loss'],
        reconstruction_config=config['reconstruction_config'],
        loss_weights=config['loss_weights'],
        loss_step=config['loss_step']
    )


    # Load weights
    model_path = Path("/home/artproject/weights/gan/GAN_V4.5_145epochs.pth")
    solver.load(model_path)

    # Testing
    solver.evaluation(data_test, save_dir=Path("/home/artproject/results"))


if __name__ == '__main__':
    import os
    import sys
    from pathlib import Path
    
    is_cuda = torch.cuda.is_available()
    print(f"Is CUDA available: {is_cuda}")
    if is_cuda:
        print(f"{torch.cuda.current_device()=}\n{torch.cuda.device_count()=}\n{torch.cuda.get_device_name()=}\n")

    parameter_root = Path("parameters")

    param_path = sys.argv[1]
    with open(parameter_root / f"{param_path}", "r") as file:
        params = json.load(file)

    
    params["device"] = "cuda"
 
    print(f"Running params: {params}")
    #run_training(params)
    run_testing(params)