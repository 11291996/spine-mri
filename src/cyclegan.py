import os
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.optim as optim


class ResidualBlock(nn.Module):
    """Residual Block without the skip connection."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(in_channels),
        )

    def forward(self, x):
        return self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()

        model = []

        # 1. c7s1-64: Reflection padding and a 7x7 conv. Changed input channels from 3 to 1.
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, 32, 7),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
        ]

        # 2. Downsampling (dk)
        model += [self.__conv_block(32, 64), self.__conv_block(64, 128)]

        # 3. Residual Blocks (Rk)
        model += [ResidualBlock(128)] * num_blocks

        # 4. Upsampling (uk)
        model += [
            self.__conv_block(128, 64, upsample=True),
            self.__conv_block(64, 32, upsample=True),
        ]

        # 5. c7s1-1: Final conv, changed output channels from 3 to 1.
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 1, 7),
        ]

        self.model = nn.Sequential(*model)

    def __conv_block(self, in_features, out_features, upsample=False):
        if upsample:
            conv = nn.ConvTranspose2d(in_features, out_features,
                                      kernel_size=3, stride=2,
                                      padding=1, output_padding=1)
        else:
            conv = nn.Conv2d(in_features, out_features,
                             kernel_size=3, stride=2, padding=1)

        return nn.Sequential(
            conv,
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    """ CycleGAN PatchGAN Discriminator """
    def __init__(self):
        super().__init__()
        in_channels = 1  # Monochrome images

        self.model = nn.Sequential(
            self.__conv_layer(in_channels, 32, norm=False),  # No InstanceNorm in first layer
            self.__conv_layer(32, 64),
            self.__conv_layer(64, 128),
            self.__conv_layer(128, 256, stride=1),  # Keep stride=1 to increase receptive field
            nn.Conv2d(256, 1, kernel_size=2, stride=1, padding=1),  # PatchGAN output
        )

    def __conv_layer(self, in_features, out_features, stride=2, norm=True):
        layers = [nn.Conv2d(in_features, out_features, kernel_size=4, stride=stride, padding=1)]
        if norm:
            layers.append(nn.InstanceNorm2d(out_features))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)  # Output is a 30x30 patch of predictions


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from .utils import MRIDataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=str, default="1")
    parser.add_argument("--data_dir", type=str, default="/data/datasets/spine/gtu/train")
    parser.add_argument("--original_modal", type=str, default="t1")
    parser.add_argument("--target_modal", type=str, default="t2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="/data/model")
    args = parser.parse_args()

    model_type = "cyclegan"

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = MRIDataset(args.data_dir, args.original_modal, args.target_modal)
    train_loader = DataLoader(dataset=train_dataset, num_workers=4, pin_memory=True, batch_size=args.batch_size, shuffle=True)

    # ========== Create Models ==========
    G = Generator(6).cuda()  # G: A → B
    F = Generator(6).cuda()  # F: B → A
    D_A = Discriminator().cuda()  # Discriminator for domain A
    D_B = Discriminator().cuda()  # Discriminator for domain B

    # ========== Apply Gaussian Weight Initialization ==========
    def weights_init_normal(m):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    G.apply(weights_init_normal)
    F.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

    # ========== Loss Functions ==========
    criterion_GAN = nn.MSELoss().cuda()  # LSGAN loss
    criterion_cycle = nn.L1Loss().cuda()  # Cycle consistency loss
    criterion_identity = nn.L1Loss().cuda()  # Identity loss (optional)

    # ========== Optimizers ==========
    g_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    f_optimizer = optim.Adam(F.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    d_A_optimizer = optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    d_B_optimizer = optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    # ========== Learning Rate Schedulers ==========
    def lambda_rule(epoch):
        if epoch < 50:
            return 1.0
        else:
            return 1.0 - (epoch - 50) / 50.0  # Linear decay after 100 epochs

    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda_rule)
    f_scheduler = torch.optim.lr_scheduler.LambdaLR(f_optimizer, lr_lambda=lambda_rule)
    d_A_scheduler = torch.optim.lr_scheduler.LambdaLR(d_A_optimizer, lr_lambda=lambda_rule)
    d_B_scheduler = torch.optim.lr_scheduler.LambdaLR(d_B_optimizer, lr_lambda=lambda_rule)

    # ========== Training Loop ==========
    num_epochs = args.num_epochs
    lambda_cycle = 10  # Cycle consistency loss weight
    lambda_identity = 5  # Identity loss weight (optional)

    for epoch in tqdm(range(num_epochs), desc='Epoch', position=0):
        for i, (real_A, real_B) in enumerate(tqdm(train_loader, desc='Batches', position=1, leave=False)):

            # ========== Preprocess Inputs ==========
            real_A = real_A.to(torch.float32).cuda()
            real_B = real_B.to(torch.float32).cuda()

            # ========== Generate Fake Images ==========
            fake_B = G(real_A)  # G(A) → Fake B
            fake_A = F(real_B)  # F(B) → Fake A

            # ========== Identity Loss (Preserves Structure) ==========
            id_A = G(real_B)  # Should be close to real_B
            id_B = F(real_A)  # Should be close to real_A
            loss_id_A = criterion_identity(id_A, real_B) * lambda_identity
            loss_id_B = criterion_identity(id_B, real_A) * lambda_identity
            loss_identity = (loss_id_A + loss_id_B) * 0.5  # Average

            # ========== GAN Loss (Adversarial Loss) ==========
            pred_fake_B = D_B(fake_B)  # D_B(G(A))
            pred_fake_A = D_A(fake_A)  # D_A(F(B))
            loss_gan_G = criterion_GAN(pred_fake_B, torch.ones_like(pred_fake_B))  # G should fool D_B
            loss_gan_F = criterion_GAN(pred_fake_A, torch.ones_like(pred_fake_A))  # F should fool D_A

            # ========== Cycle Consistency Loss ==========
            cycle_A = F(fake_B)  # F(G(A)) → Should be close to real_A
            cycle_B = G(fake_A)  # G(F(B)) → Should be close to real_B
            loss_cycle_A = criterion_cycle(cycle_A, real_A) * lambda_cycle
            loss_cycle_B = criterion_cycle(cycle_B, real_B) * lambda_cycle
            loss_cycle = (loss_cycle_A + loss_cycle_B) * 0.5  # Average

            # ========== Total Generator Loss ==========
            loss_G = loss_gan_G + loss_gan_F + loss_cycle + loss_identity

            # ========== Update Generators ==========
            G.zero_grad()
            F.zero_grad()
            loss_G.backward()
            g_optimizer.step()
            f_optimizer.step()

            # ========== Train Discriminators ==========
            # Train D_A
            pred_real_A = D_A(real_A)
            pred_fake_A = D_A(fake_A.detach())
            loss_d_real_A = criterion_GAN(pred_real_A, torch.ones_like(pred_real_A))
            loss_d_fake_A = criterion_GAN(pred_fake_A, torch.zeros_like(pred_fake_A))
            loss_D_A = (loss_d_real_A + loss_d_fake_A) * 0.5  # Divide by 2

            D_A.zero_grad()
            loss_D_A.backward()
            d_A_optimizer.step()

            # Train D_B
            pred_real_B = D_B(real_B)
            pred_fake_B = D_B(fake_B.detach())
            loss_d_real_B = criterion_GAN(pred_real_B, torch.ones_like(pred_real_B))
            loss_d_fake_B = criterion_GAN(pred_fake_B, torch.zeros_like(pred_fake_B))
            loss_D_B = (loss_d_real_B + loss_d_fake_B) * 0.5  # Divide by 2

            D_B.zero_grad()
            loss_D_B.backward()
            d_B_optimizer.step()

        # Print Progress
        print(f"Epoch [{epoch+1}/{num_epochs}] | G Loss: {loss_G.item():.4f} | D_A Loss: {loss_D_A.item():.4f} | D_B Loss: {loss_D_B.item():.4f}")

        # Update Learning Rate
        g_scheduler.step()
        f_scheduler.step()
        d_A_scheduler.step()
        d_B_scheduler.step()

    subject_name = args.data_dir.split('/')[-3]
    dataset_name = args.data_dir.split('/')[-2]

    if not os.path.exists(os.path.join(args.save_dir, dataset_name)):
        os.makedirs(os.path.join(args.save_dir, model_type, subject_name, dataset_name))

    # save model
    torch.save(G.state_dict(), os.path.join(args.save_dir, model_type, subject_name, dataset_name, f'G_{args.original_modal}_{args.target_modal}.pth'))
    torch.save(F.state_dict(), os.path.join(args.save_dir, model_type, subject_name, dataset_name, f'D_{args.original_modal}_{args.target_modal}.pth'))
    torch.save(D_A.state_dict(), os.path.join(args.save_dir, model_type, subject_name, dataset_name, f'D_A_{args.original_modal}_{args.target_modal}.pth'))
    torch.save(D_B.state_dict(), os.path.join(args.save_dir, model_type, subject_name, dataset_name, f'D_B_{args.original_modal}_{args.target_modal}.pth'))

    print("Training Finished!")
