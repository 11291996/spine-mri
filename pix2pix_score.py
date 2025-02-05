import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from pix2pix import BrainDataset, Generator
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

def torch_psnr(img1, img2):
    """ Compute PSNR in PyTorch """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming input is normalized [0,1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def torch_mae(img1, img2):
    """ Compute MAE in PyTorch """
    #normalize to 0-1
    img1 = img1 / 255
    img2 = img2 / 255
    return torch.mean(torch.abs(img1 - img2)).item()

def torch_ssim(img1, img2):
    """ Compute Structural Similarity Index (SSIM) in PyTorch without skimage """
    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)

    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    sigma1 = torch.var(img1)
    sigma2 = torch.var(img2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))

    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2))
    return ssim.item()


def unnormalize(tensor):
    """ Unnormalize tensor from [-1, 1] to [0, 1] """
    return (tensor * 0.5) + 0.5  # Reverse normalization

if __name__ == "__main__":

    # # Load images using PIL
    # img1 = Image.open("output.png").convert("L")  # Convert to grayscale
    # img2 = Image.open("output2.png").convert("L")

    # # Convert images to Torch tensors
    # transform = transforms.ToTensor()
    # img1_torch = transform(img1).unsqueeze(0)  # Add batch dimension
    # img2_torch = transform(img2).unsqueeze(0)

    # # Compute metrics
    # psnr_value = torch_psnr(img1_torch, img2_torch)
    # ssim_value = torch_ssim(img1_torch, img2_torch)
    # mae_value = torch_mae(img1_torch, img2_torch)

    # # Print results
    # print(f"PSNR: {psnr_value:.2f} dB")
    # print(f"SSIM: {ssim_value:.4f}")
    # print(f"MAE: {mae_value:.4f}")

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    test_dataset = BrainDataset("/data/brain/BraTS2020/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData", "t1", "t1ce")
    test_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=1, shuffle=False)

    # Load the pre-trained model
    G = Generator().cuda()
    G.load_state_dict(torch.load('/data/model/pix2pix/G.pth'))
    G.eval()

    # Track overall metrics
    total_psnr, total_ssim, total_mae, count = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for i, (original_images, target_images) in enumerate(tqdm(test_loader)):

            for slice in range(original_images.shape[4]):

                real_a = original_images[:, :, :, :, slice]  # (B, C, H, W)
                real_b = target_images[:, :, :, :, slice]

                # Transform & Move to GPU
                real_a = test_dataset.transform(real_a).to(torch.float32).cuda()
                real_b = test_dataset.transform(real_b).to(torch.float32).cuda()

                # Generate fake images
                fake_b = G(real_a)

                if slice == original_images.shape[4] // 2:

                    # squeeze the batch dimension and channel dimension
                    fake_b = fake_b.squeeze(0).squeeze(0)
                    real_b = real_b.squeeze(0).squeeze(0)

                    #save the image
                    fake_b = unnormalize(fake_b)
                    real_b = unnormalize(real_b)

                    plt.imshow(fake_b.cpu().numpy(), cmap="gray")
                    plt.savefig('output_fake3.png')
                    plt.imshow(real_b.cpu().numpy(), cmap="gray")
                    plt.savefig('output_real3.png')

                    exit()

                fake_b = unnormalize(fake_b)
                real_b = unnormalize(real_b)

                # Compute metrics
                psnr_value = torch_psnr(fake_b, real_b)
                ssim_value = torch_ssim(fake_b, real_b)
                mae_value = torch_mae(fake_b, real_b)

                # Accumulate metrics
                total_psnr += psnr_value
                total_ssim += ssim_value
                total_mae += mae_value
                count += 1

                # print(f"Slice {slice}: PSNR={psnr_value:.2f}, SSIM={ssim_value:.4f}, MAE={mae_value:.4f}")

        # Compute average metrics
        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count
        avg_mae = total_mae / count

        print(f"\n🔹 **Final Test Metrics**")
        print(f"✅ Average PSNR: {avg_psnr:.2f} dB")
        print(f"✅ Average SSIM: {avg_ssim:.4f}")
        print(f"✅ Average MAE: {avg_mae:.4f}")