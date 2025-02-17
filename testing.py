from datasets import EvaluationDataset
from torch.utils.data import DataLoader
from helper import save_evaluation
import os
import torch
from AttentionUNet import AttentionUnet
import tqdm
import torch.nn.functional as F

working_dir = os.getcwd()
best_model_path = "models/Attention_Unet_Aug_l_100eps.pth"

evaluation_data = EvaluationDataset(r"/content/drive/MyDrive/Colab Notebooks/DLCV/Project/damaged_resized")

data_loader = DataLoader(evaluation_data, batch_size=4, shuffle=False, num_workers=2)

# create model
##### pathnames here ######
model_path = os.path.join(working_dir, best_model_path)
###########################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = AttentionUnet(in_channels=3, out_channels=1, filters=[32, 64, 128, 256]).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()

with torch.no_grad():
  for images, image_names in tqdm.tqdm(data_loader):
    plot_dir = "evaluations/AugAttUNet_l_100eps"
    os.makedirs(os.path.join(working_dir, plot_dir), exist_ok=True)

    # transfer to device
    images = images.to(device)

    # get the predictions
    preds = model(images)

    # convert logits to binary
    threshold = 0.5
    preds_bw = (F.sigmoid(preds)>threshold).long()

    # save the evaluation images under ./evaluation_real_data
    for i, image_name in enumerate(image_names):
      path = os.path.join(working_dir, f"{plot_dir}/{image_name}")
      save_evaluation(images[i], preds_bw[i], path)