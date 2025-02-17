import matplotlib.pyplot as plt

def get_model_parameters(m):
    total_params = sum(
        param.numel() for param in m.parameters()
    )
    return total_params

def print_model_parameters(m):
    num_model_parameters = get_model_parameters(m)
    print(f"The Model has {num_model_parameters/1e6:.2f}M parameters")


def save_image(image, prediction, mask, epoch, step, plot_dir):
  image = image.permute(1,2,0)
  mask = (mask*255).permute(1,2,0)
  prediction = (prediction*255).permute(1,2,0)

  fig, axs = plt.subplots(1, 3, figsize=(15,5))

  axs[0].imshow(image.cpu())
  axs[0].set_title("Damaged Image")
  axs[0].axis('off')

  axs[1].imshow(mask.cpu())
  axs[1].set_title("Mask")
  axs[1].axis('off')

  axs[2].imshow(prediction.cpu())
  axs[2].set_title("Learnt Mask")
  axs[2].axis('off')

  plt.savefig(f"{plot_dir}/epoch_{epoch+1}_step_{step}.png")
  plt.close()

def save_evaluation(image, prediction, path):
  image = image.permute(1,2,0)
  prediction = (prediction*255).permute(1,2,0)

  fig, axs = plt.subplots(1, 2, figsize=(10,5))

  axs[0].imshow(image.cpu())
  axs[0].set_title("Damaged Image")
  axs[0].axis('off')

  axs[1].imshow(prediction.cpu())
  axs[1].set_title("Predicted Mask")
  axs[1].axis('off')

  plt.savefig(path)
  plt.close()