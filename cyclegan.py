from torch import nn
import torch
from pix2pix import BrainDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tqdm import tqdm

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.InstanceNorm2d(256),
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(1, 64, 7),  # Input channels changed from 3 to 1
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        model += [self.__conv_block(64, 128), self.__conv_block(128, 256)]

        # Residual Blocks
        model += [ResidualBlock()] * num_blocks

        # Upsampling
        model += [
            self.__conv_block(256, 128, upsample=True),
            self.__conv_block(128, 64, upsample=True),
        ]

        # Output layer
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, 1, 7), nn.Tanh()]  # Outputs RGB

        self.model = nn.Sequential(*model)

    def __conv_block(self, in_features, out_features, upsample=False):
        if upsample:
            conv = nn.ConvTranspose2d(in_features, out_features, 3, 2, 1, output_padding=1)
        else:
            conv = nn.Conv2d(in_features, out_features, 3, 2, 1)

        return nn.Sequential(
            conv,
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            self.__conv_layer(1, 64, norm=False),  # Input channels changed from 3 to 1
            self.__conv_layer(64, 128),
            self.__conv_layer(128, 256),
            self.__conv_layer(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def __conv_layer(self, in_features, out_features, stride=2, norm=True):
        layer = [nn.Conv2d(in_features, out_features, 4, stride, 1)]
        if norm:
            layer.append(nn.InstanceNorm2d(out_features))
        layer.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layer)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # ========== Load Dataset ==========
    train_dataset = BrainDataset("/data/brain/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData", "t1", "t1ce")
    train_loader = DataLoader(dataset=train_dataset, num_workers=4, pin_memory=True, batch_size=30, shuffle=True)  # Smaller batch size for stability

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
    g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    f_optimizer = optim.Adam(F.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_A_optimizer = optim.Adam(D_A.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_B_optimizer = optim.Adam(D_B.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # ========== Learning Rate Schedulers ==========
    def lambda_rule(epoch):
        if epoch < 100:
            return 1.0
        else:
            return 1.0 - (epoch - 100) / 100.0  # Linear decay after 100 epochs

    g_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=lambda_rule)
    f_scheduler = torch.optim.lr_scheduler.LambdaLR(f_optimizer, lr_lambda=lambda_rule)
    d_A_scheduler = torch.optim.lr_scheduler.LambdaLR(d_A_optimizer, lr_lambda=lambda_rule)
    d_B_scheduler = torch.optim.lr_scheduler.LambdaLR(d_B_optimizer, lr_lambda=lambda_rule)

    # ========== Training Loop ==========
    num_epochs = 200
    lambda_cycle = 10  # Cycle consistency loss weight
    lambda_identity = 5  # Identity loss weight (optional)

    for epoch in tqdm(range(num_epochs)):
        for i, (original_images, target_images) in enumerate(train_loader):

            for slice in range(original_images.shape[4]):
                real_A = original_images[:, :, :, :, slice]
                real_B = target_images[:, :, :, :, slice]

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
                print(loss_G)
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

        # Update Learning Rate
        g_scheduler.step()
        f_scheduler.step()
        d_A_scheduler.step()
        d_B_scheduler.step()

        # Print Progress
        print(f"Epoch [{epoch+1}/{num_epochs}] | G Loss: {loss_G.item():.4f} | D_A Loss: {loss_D_A.item():.4f} | D_B Loss: {loss_D_B.item():.4f}")

    # Save model every 50 epochs
    torch.save(G.state_dict(), f'/data/model/cyclegan/G.pth')
    torch.save(F.state_dict(), f'/data/model/cyclegan/F.pth')
    torch.save(D_A.state_dict(), f'/data/model/cyclegan/D_A.pth')
    torch.save(D_B.state_dict(), f'/data/model/cyclegan/D_B.pth')

    print("Training Finished!")
