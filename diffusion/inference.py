import os
import torch
import matplotlib.pyplot as plt
from model import UNet, Diffusion
import cv2
import numpy as np
from torchvision import transforms
def load_image_and_mask(img_path, mask_path):
    transform = transforms.ToTensor()  # Ensures (C, H, W) for images

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0)  # Converts to (1,C, H, W)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = torch.tensor(mask, dtype=torch.float32) / 255.0  # Normalize
    mask = mask.unsqueeze(0).unsqueeze(0)  # Ensure (1,1, H, W)

    return img, mask

def inference(model_path, img_path, mask_path, timesteps=500, output_dir="./inpainted_output"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    diffusion = Diffusion(timesteps=timesteps).to(device)

    image, mask = load_image_and_mask(img_path, mask_path)
    image, mask = image.to(device), mask.to(device)
    print(image.shape,mask.shape)
    x_t = image.clone()
    with torch.no_grad():
        for t in reversed(range(timesteps)):
            t_tensor = torch.full((x_t.shape[0],), t, device=device, dtype=torch.float32) / timesteps
            t_tensor = t_tensor.view(-1, 1, 1, 1).expand(-1, 1, x_t.shape[2], x_t.shape[3]) 

            input_t = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)

            torch.cuda.empty_cache()

            input_tensor = torch.cat([x_t, mask, t_tensor], dim=1)

            predicted_noise = model(input_tensor)


            if t % 20 == 1:
                pred_noise_img = predicted_noise[0].permute(1, 2, 0).detach().cpu().numpy()
                x_t_img = x_t[0].permute(1, 2, 0).detach().cpu().numpy()

                plt.figure(figsize=(10,5))
                plt.subplot(1,2,1)
                plt.imshow(np.clip(pred_noise_img, -1, 1))  
                plt.title(f"Predicted Noise at t={t}")
                plt.axis("off")

                plt.subplot(1,2,2)
                plt.imshow(np.clip(x_t_img, 0, 1))
                plt.title(f"Noisy Image at t={t}")
                plt.axis("off")

                plt.savefig(f"./debug/noise_t_{t}.png")
                plt.close()

    os.makedirs(output_dir, exist_ok=True)
    save_inpainted_image(image, x_t, mask, output_dir)

def save_inpainted_image(original, inpainted, mask, output_dir):
    original, inpainted, mask = original.cpu(), inpainted.cpu(), mask.cpu()
    original = original[0].permute(1, 2, 0).numpy()
    inpainted = inpainted[0].permute(1, 2, 0).numpy()
    mask = mask[0, 0].numpy()
    
    # Plot the images for comparison
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(original)
    axs[0].set_title("Original Image")
    axs[0].axis("off")
    
    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Mask")
    axs[1].axis("off")
    
    axs[2].imshow(inpainted)
    axs[2].set_title("Inpainted Image")
    axs[2].axis("off")
    
    # Save the result
    plt.savefig(f"{output_dir}/result.png")
    plt.close()

if __name__ == "__main__":
    inference("./models/best_model.pth", "./data/damaged/1.jpg", "./data/mask/1.jpg")
