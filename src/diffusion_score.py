from .diffusion import DDIMUNet, DDIMSampler
from .gan_score import torch_psnr, torch_ssim_masked, torch_mae, torch_lpips
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import os
import lpips
import numpy
from PIL import Image
from torch.utils.data import DataLoader
from .utils import MRIDataset
import argparse
from .logger import TrainLogger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=str, default="1")
    parser.add_argument("--model_dir", type=str, default="/data/model/diffusion/spine/gtu/model_t1_t2.pth")
    parser.add_argument("--datasets_dir", type=str, default="/data/datasets/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    original_modal = args.model_dir.split("/")[-1].split("_")[1]
    target_modal = args.model_dir.split("/")[-1].split("_")[2].split(".")[0]

    subject_name = args.model_dir.split("/")[-3]
    dataset_name = args.model_dir.split("/")[-2]

    data_dir = os.path.join(args.datasets_dir, subject_name, dataset_name, "test")

    save_dir = args.model_dir.split("/")[:-1]
    save_dir = '/'.join(save_dir)
    logger = TrainLogger(log_dir=save_path, prefix=f"score_{original_modal}_{target_modal}")

    test_dataset = MRIDataset(data_dir, original_modal, target_modal)
    # from torch.utils.data import random_split
    # train_size = int(0.1 * len(test_dataset))
    # val_size = len(test_dataset) - train_size 
    # test_dataset, _ = random_split(test_dataset, [train_size, val_size])

    test_loader = DataLoader(dataset=test_dataset, num_workers=4, pin_memory=True, batch_size=args.batch_size, shuffle=False)

    # Initialize model and sampler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DDIMUNet(image_size=256, in_channels=2,
                    model_channels=96, out_channels=2, 
                    num_res_blocks=1, attention_resolutions=[32,16,8],
                    channel_mult=[1, 2, 4, 8]).to(device)
    model.load_state_dict(torch.load(args.model_dir))
    sampler = DDIMSampler(model, num_ddim_steps=50, eta=0.0)  # Use deterministic DDIM

    psnr, ssim, mae, lpips_score, count = 0.0, 0.0, 0.0, 0.0, 0

    loss_fn = lpips.LPIPS(net="alex").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Generate one random noisy image
    for i, (original_images, target_images) in enumerate(test_loader):
        original_images = original_images.to(device, non_blocking=True)
        target_images = target_images.to(device, non_blocking=True)

        timesteps = torch.randint(0, 1000, (1,), device=device).float()
        rand_noise = torch.randn_like(original_images)
        input_images = torch.cat([original_images, rand_noise], dim=1)

        pred_target = sampler.sample(input_images, timesteps)

        pred_target = pred_target[:, 1].unsqueeze(1)

        target_images = target_images.squeeze().squeeze()
        pred_target = pred_target.squeeze().squeeze()

        sampled_images = (pred_target - pred_target.min()) / (pred_target.max() - pred_target.min())
        target_images = (target_images - target_images.min()) / (target_images.max() - target_images.min())

        # Compute metrics
        psnr += torch_psnr(sampled_images, target_images)
        ssim += torch_ssim_masked(sampled_images, target_images)
        mae += torch_mae(sampled_images, target_images)
        lpips_score += torch_lpips(sampled_images.cuda(), target_images.cuda(), loss_fn)
        count += 1
        
    logger.log(f"Average PSNR: {psnr / count:.4f} dB")
    logger.log(f"Average SSIM: {ssim / count:.4f}")
    logger.log(f"Average MAE: {mae / count:.6f}")
    logger.log(f"Average LPIPS: {lpips_score / count:.4f}")
