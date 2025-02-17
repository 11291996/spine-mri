import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import os
from .utils.model import CUNet
from .utils.mdn_model import SimpleMLP
import torch.optim as optim
from ..utils import MRIDataset
from ..diffusion import get_alphas_cumprod
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils.sampler import CDMDDIMSampler
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--device_num", type=str, default="1")
parser.add_argument("--data_dir", type=str, default="/data/datasets/spine/gtu/train")
parser.add_argument("--original_modal", type=str, default="t1")
parser.add_argument("--target_modal", type=str, default="t2")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--beta2", type=float, default=0.999)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--save_dir", type=str, default="/data/model")

model_type = "cdm"

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

subject_name = args.data_dir.split("/")[-3]
dataset_name = args.data_dir.split("/")[-2]

# Test the model
model = CUNet(image_size=256, in_channels=1,
                  model_channels=96, out_channels=1, 
                  num_res_blocks=1, attention_resolutions=[32,16,8],
                  channel_mult=[1, 1, 2, 2]).cuda()

mdn_model = SimpleMLP(in_channels=192, time_embed_dim= 192, model_channels=1536, bottleneck_channels=1536, out_channels=192, num_res_blocks=12)

mdn_model_path = os.path.join(args.save_dir, model_type, subject_name, dataset_name, f'mdn_model_{args.original_modal}_{args.target_modal}.pth')

# #load the encoder weights
try:
    mdn_model.load_state_dict(torch.load(mdn_model_path))
except:
    print("MDN model not found. Train mrm model first, or check the save directory.")
    exit()

train_dataset = MRIDataset(args.data_dir, args.original_modal, args.target_modal)
train_loader = DataLoader(dataset=train_dataset, num_workers=4, pin_memory=True, batch_size=args.batch_size, shuffle=True)

lr = args.lr
beta1 = args.beta1
beta2 = args.beta2

opt = optim.Adam(list(model.parameters()) + list(mdn_model.parameters()), lr=lr, betas=(beta1, beta2))
model.train()
mdn_model.eval()

model.to(device)
mdn_model.to(device)

sampler = CDMDDIMSampler(model, mdn_model, num_ddim_steps=50, eta=0.0)

num_epochs = args.num_epochs

import matplotlib.pyplot as plt

for epoch in tqdm(range(num_epochs), desc='Epoch', position=0):
    for i, (original_images, target_images) in enumerate(tqdm(train_loader, desc='Batches', position=1, leave=False)):

        original_images = original_images.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)

        opt.zero_grad()

        batch_size = original_images.shape[0]

        #time step for diffusion
        time_steps = torch.randint(0, 1000, (batch_size, ), device=device).float()
        #create noise for the target condtion size (batch_size, 192)
        noise = torch.randn(batch_size, 192, device=device)
        pred_cond = mdn_model(noise, time_steps)

        # image_noise = torch.randn_like(original_images)
        # alpha_t = 0.9
        # original_images = original_images * alpha_t + image_noise * (1 - alpha_t)

        pred_target = model(original_images, pred_cond)

        #use L2 loss
        loss = torch.nn.functional.mse_loss(pred_target, target_images)

        opt.zero_grad()
        loss.backward()
        opt.step()

    if (epoch + 1) % 10 == 0 and epoch != 0:

    #     random_cond = torch.randn(1, 192, device=device)
    #     sampled_images = sampler.sample(original_images, random_cond)

    #     #save the image in a single plot
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(8, 3))  # Reduce figure size to make it more compact

    #     original_images = original_images[0, 0]
    #     target_images = target_images[0, 0]
    #     sampled_images = sampled_images[0, 0]

    #     # Plot Original Image
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(original_images.cpu().numpy(), cmap='gray')
    #     plt.axis("off")
    #     plt.title("Original", fontsize=10)

    #     # Plot Target Image
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(target_images.cpu().numpy(), cmap='gray')
    #     plt.axis("off")
    #     plt.title("Target", fontsize=10)

    #     # Plot Denoised Image
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(sampled_images.cpu().detach().numpy(), cmap='gray')
    #     plt.axis("off")
    #     plt.title("Denoised", fontsize=10)

    #     # Adjust layout for compact spacing
    #     plt.subplots_adjust(wspace=0.05, hspace=0)  # Reduce spacing between images
    #     plt.tight_layout(pad=0)  # Remove extra padding

    #     # Save the image with minimal margin
    #     plt.savefig(f"cdm_output_{epoch}.png", bbox_inches='tight', dpi=300)

        print(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")

torch.save(model.state_dict(), os.path.join(args.save_dir, model_type, subject_name, dataset_name, f'model_{args.original_modal}_{args.target_modal}.pth'))