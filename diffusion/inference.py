import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import argparse
from model import UNet
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Total diffusion timesteps
T = 500
betas = torch.linspace(0.0001, 0.02, T).to(device)  
alphas = 1 - betas  
alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)

def q_sample(x_0, t, noise):
    """
    Add noise to image x_0 at timestep t.
    Formula: q(x_t | x_0) = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
    """
    sqrt_alpha_cumprod_t = torch.sqrt(alphas_cumprod[t]).view(1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).view(1, 1, 1, 1)
    
    return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

def load_image(image_path, mask_path, ground_truth_path, image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Assuming same normalization as training
    ])
    
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    ground_truth = Image.open(ground_truth_path).convert("RGB")
    
    image = transform(image).to(device)
    mask = transform(mask).to(device)
    ground_truth = transform(ground_truth).to(device)
    
    mask = (mask > 0.5).float()

    # Replace masked pixels with white (assuming normalized range [-1, 1], so white is 1)
    image = (1 - mask) * image + mask

    return image, mask, ground_truth


@torch.no_grad()
def repaint_denoise(model, xt, x0, mask, betas, alphas, alphas_cumprod, num_repaint_steps=5):
    """
    Implements the RePaint inpainting approach.
    """
    T = len(betas)
    debug_folder = "debug_images"
    os.makedirs(debug_folder, exist_ok=True)

    for t in range(T - 1, 0, -1):  # Iterate from T to 1
        t_tensor = torch.tensor([t], device=xt.device, dtype=torch.long)
        noise = torch.randn_like(xt)
        for _ in range(num_repaint_steps):  # Refinement steps
            noise_pred = model(xt, t_tensor)  # Predict noise
            x_known_t_minus_1 = q_sample(x0, t-1, noise)

            beta_t = betas[t].view(1, 1, 1, 1)
            alpha_t = alphas[t].view(1, 1, 1, 1)
            sigma_t = torch.sqrt(beta_t)

            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1 - alphas_cumprod[t]).view(1, 1, 1, 1)

            z = torch.randn_like(xt) if t > 1 else 0  # Only add noise if t > 1

            x_unknown_t_minus_1 = (1 / torch.sqrt(alpha_t)) * (
                xt - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred
            ) + sigma_t * z  

            xt_minus_1 = (1-mask) * x_known_t_minus_1 + mask * x_unknown_t_minus_1
            if t % 10 == 0:
                first = (1-mask) * x_known_t_minus_1
                second = mask * x_unknown_t_minus_1
            if t > 1:  
                beta_t_minus_1 = betas[t-1].view(1, 1, 1, 1)
                noise = torch.randn_like(xt)
                xt = torch.sqrt(1 - beta_t_minus_1) * xt_minus_1 + torch.sqrt(beta_t_minus_1) * noise

        torch.cuda.empty_cache()  # Free memory
        if t%10 == 0:
            print(t)

    return xt

@torch.no_grad()
def inpaint(image, mask, model, betas, alphas, alphas_cumprod, num_repaint_steps, device):
    model.eval()

    image = image.unsqueeze(0).to(device)  # Add batch dimension
    mask = mask.unsqueeze(0).to(device)  # Add batch dimension
    x_t = torch.randn_like(image, dtype=torch.float32).to(device)

    final_result = repaint_denoise(model, x_t, image, mask, betas, alphas, alphas_cumprod, num_repaint_steps)

    return final_result

def save_image(tensor, filename):
    tensor = tensor.squeeze(0).cpu().detach()
    tensor = (tensor + 1) / 2  # Normalize back from [-1, 1] to [0, 1]
    tensor = transforms.ToPILImage()(tensor.clamp(0, 1))
    tensor.save(filename)

def save_plot(input_image, denoised_image, ground_truth_image,save_path):
    input_image = (input_image + 1) / 2  # Convert from [-1, 1] to [0, 1]
    denoised_image = (denoised_image + 1) / 2
    ground_truth_image = (ground_truth_image + 1) / 2

    input_image = input_image.squeeze(0).cpu().detach().clamp(0, 1)
    denoised_image = denoised_image.squeeze(0).cpu().detach().clamp(0, 1)
    ground_truth_image = ground_truth_image.squeeze(0).cpu().detach().clamp(0, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(input_image.permute(1, 2, 0))
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(denoised_image.permute(1, 2, 0))
    axes[1].set_title("Denoised Image")
    axes[1].axis("off")

    axes[2].imshow(ground_truth_image.permute(1, 2, 0))
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--mask", type=str, required=True, help="Path to binary mask")
    parser.add_argument("--ground_truth", type=str, required=True, help="Path to ground truth image")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--output", type=str, default="output.png", help="Path to save result")
    parser.add_argument("--num_repaint_steps", type=int, default=10, help="Number of repaint steps")
    parser.add_argument("--image_size", type=int, default=256, help="Size of input image")
    args = parser.parse_args()

    # Load model
    model = UNet()
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # Load image, mask, and ground truth
    image, mask, ground_truth = load_image(args.image, args.mask, args.ground_truth, args.image_size)

    result = inpaint(image, mask, model, betas, alphas, alphas_cumprod, args.num_repaint_steps, device)

    save_image(result, args.output)
    print(f"Final output saved as {args.output}")

    # Save the plot
    save_plot(image, result, ground_truth, "result.png")
