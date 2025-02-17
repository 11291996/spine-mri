import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from .cyclegan import Generator
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import lpips

def torch_psnr(img1, img2):
    """ Compute PSNR in PyTorch """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming input is normalized [0,1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def torch_mae(img1, img2, precision=10):
    """ Compute MAE in PyTorch with higher precision """
    mae = torch.mean(torch.abs(img1 - img2)).item()
    return round(mae, precision)  

def torch_ssim(img1, img2):
    """ Compute Structural Similarity Index (SSIM) in PyTorch without skimage """
    C1 = (0.001 ** 2)
    C2 = (0.001 ** 2)

    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    sigma1 = torch.var(img1)
    sigma2 = torch.var(img2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim.item()

def torch_ssim_masked(img1, img2):
    """ Compute SSIM in PyTorch after removing black areas """

    # Define a small constant to avoid division by zero
    C1 = (0.001 ** 2)
    C2 = (0.001 ** 2)

    # Create a mask to ignore black areas (assuming black is 0)
    mask = (img1 > 0) & (img2 > 0)  # Consider only non-black pixels in both images

    # Ensure there are valid pixels to compare
    if mask.sum() == 0:
        print("Warning: No non-black pixels found in both images!")
        return float('nan')

    # Filter out black pixels
    img1_masked = img1[mask]
    img2_masked = img2[mask]

    # Compute mean and variance only on non-black pixels
    mu1 = torch.mean(img1_masked)
    mu2 = torch.mean(img2_masked)
    sigma1 = torch.var(img1_masked)
    sigma2 = torch.var(img2_masked)
    sigma12 = torch.mean((img1_masked - mu1) * (img2_masked - mu2))

    # Compute SSIM
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    
    return ssim.item()

def torch_lpips(img1, img2, loss_fn):
    """
    Compute LPIPS between img1 and img2.
    
    img1 and img2 should be PyTorch tensors of shape [B, C, H, W] 
    with values normalized to [-1, 1] (which is the convention for LPIPS).
    """
    # LPIPS returns a tensor of shape [B, 1, 1, 1]. We can take the mean over the batch.
    lpips_value = loss_fn.forward(img1, img2)
    return lpips_value.mean().item()

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from .utils import MRIDataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=str, default="1")
    parser.add_argument("--model_dir", type=str, default="/data/model/pix2pix/spine/gtu/G_t1_t2.pth")
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

    test_dataset = MRIDataset(data_dir, original_modal, target_modal)
    test_loader = DataLoader(dataset=test_dataset, num_workers=4, pin_memory=True, batch_size=args.batch_size, shuffle=False)

    # Load the pre-trained model
    eval_mode = args.model_dir.split("/")[-4]

    if eval_mode == "pix2pix":
        G = GeneratorUNet().cuda()
        G.load_state_dict(torch.load(args.model_dir))
        G.eval()
    elif eval_mode == "cyclegan":
        G = Generator(6).cuda()
        G.load_state_dict(torch.load(args.model_dir))
        G.eval()

    # Track overall metrics
    total_psnr, total_ssim, total_mae, total_lpips, count = 0.0, 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for i, (original_images, target_images) in enumerate(tqdm(test_loader)):

            real_a = original_images.cuda()
            real_b = target_images.cuda()

            # Generate fake images
            fake_b = G(real_a)

            # Save the images
            fake_b = fake_b[0, :, :, :]
            real_b = real_b[0, :, :, :]

            fake_b = fake_b.squeeze(0)
            real_b = real_b.squeeze(0)

            # Normalize the images to [0, 1]
            fake_b = (fake_b - fake_b.min()) / (fake_b.max() - fake_b.min())
            real_b = (real_b - real_b.min()) / (real_b.max() - real_b.min())

            # Compute metrics
            psnr_value = torch_psnr(fake_b, real_b)
            ssim_value = torch_ssim_masked(fake_b, real_b)
            mae_value = torch_mae(fake_b, real_b)
            lpips_value = torch_lpips(fake_b.cuda(), real_b.cuda(), loss_fn=lpips.LPIPS(net="alex").to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')))

            # Accumulate metrics
            total_psnr += psnr_value
            total_ssim += ssim_value
            total_mae += mae_value
            total_lpips += lpips_value
            count += 1

            print(f"PSNR: {psnr_value:.2f} dB")
            print(f"SSIM: {ssim_value:.4f}")
            print(f"MAE: {mae_value:.4f}")
            print(f"LPIPS: {lpips_value:.4f}")

    # Compute average metrics
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    avg_mae = total_mae / count
    avg_lpips = total_lpips / count

    # Print results
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average MAE: {avg_mae:.6f}")
    print(f"Average LPIPS: {avg_lpips:.4f}")
        