from .utils.sampler import CDMDDIMSampler
from .utils.mdn_model import SimpleMLP
from .utils.model import CUNet
import os 
from ..gan_score import torch_psnr, torch_ssim_masked, torch_mae, torch_lpips
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import lpips
import numpy
from PIL import Image
from ..utils import MRIDataset
import argparse
from ..logger import TrainLogger

parser = argparse.ArgumentParser()
parser.add_argument("--device_num", type=str, default="1")
parser.add_argument("--model_dir", type=str, default="/data/model/diffusion/spine/gtu/model_t1_t2.pth")
parser.add_argument("--datasets_dir", type=str, default="/data/datasets/")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--beta1", type=float, default=0.5)
parser.add_argument("--beta2", type=float, default=0.999)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CUNet(image_size=256, in_channels=1,
                  model_channels=96, out_channels=1, 
                  num_res_blocks=1, attention_resolutions=[32,16,8],
                  channel_mult=[1, 1, 2, 2]).cuda()
mdn_model = SimpleMLP(in_channels=192, time_embed_dim= 192, model_channels=1536, bottleneck_channels=1536, out_channels=192, num_res_blocks=12).cuda()

original_modal = args.model_dir.split("/")[-1].split("_")[1]
target_modal = args.model_dir.split("/")[-1].split("_")[2].split(".")[0]

subject_name = args.model_dir.split("/")[-3]
dataset_name = args.model_dir.split("/")[-2]

data_dir = os.path.join(args.datasets_dir, subject_name, dataset_name, "test")
 
test_dataset = MRIDataset(data_dir, original_modal, target_modal)
test_loader = DataLoader(dataset=test_dataset, num_workers=4, pin_memory=True, batch_size=args.batch_size, shuffle=False)

# Initialize model and sampler
model.load_state_dict(torch.load(args.model_dir))

save_dir = args.model_dir.split("/")[:-4]
save_dir = '/'.join(save_dir)

model_type = 'cdm'
logger = TrainLogger(os.path.join(save_dir, model_type, subject_name, dataset_name), prefix=f"score_{original_modal}_{target_modal}")
   

mdn_model_path = os.path.join(save_dir, model_type, subject_name, dataset_name, f'mdn_{original_modal}_{target_modal}.pth')
mdn_model.load_state_dict(torch.load(mdn_model_path))

sampler = CDMDDIMSampler(model, mdn_model, num_ddim_steps=50, eta=0.0)

psnr, ssim, mae, lpips_score, count = 0.0, 0.0, 0.0, 0.0, 0

loss_fn = lpips.LPIPS(net="alex").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Generate one random noisy image
for i, (original_images, target_images) in enumerate(test_loader):
    
    original_images = original_images.to(device, non_blocking=True)
    target_images = target_images.to(device, non_blocking=True)

    random_cond = torch.randn(1, 192, device=device)

    sampled_images = sampler.sample(original_images, random_cond)

    # Visualize the original, noisy, and denoised images
    # detach() is used to prevent backpropagation through the visualization code
    # squeeze two dimensions to get a 2D image 
    sampled_images = sampled_images[:, 0]
    target_images = target_images.squeeze(0)

    #covert the sampled images to [0, 1] using min-max normalization relative to the entire dataset
    sampled_images = (sampled_images - sampled_images.min()) / (sampled_images.max() - sampled_images.min())
    target_images = (target_images - target_images.min()) / (target_images.max() - target_images.min())

    # Compute metrics
    psnr += torch_psnr(sampled_images, target_images)
    ssim += torch_ssim_masked(sampled_images, target_images)
    mae += torch_mae(sampled_images, target_images)
    lpips_score += torch_lpips(sampled_images.cuda(), target_images.cuda(), loss_fn)
    count += 1

    # print(f"Average PSNR: {psnr / count:.4f}")
    # print(f"Average SSIM: {ssim / count:.4f}")
    # print(f"Average MAE: {mae / count:.6f}")
    # print(f"Average LPIPS: {lpips_score / count:.4f}")

logger.log(f"Average PSNR: {psnr / count:.4f}")
logger.log(f"Average SSIM: {ssim / count:.4f}")
logger.log(f"Average MAE: {mae / count:.6f}")
logger.log(f"Average LPIPS: {lpips_score / count:.4f}")
