import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import nibabel as nib
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from math import log10 # For metric function
import os

# Load Dataset from ImageFolder
class BrainDataset(data.Dataset):
    def __init__(self, mri_dir, original_modal, target_modal):
        super(BrainDataset, self).__init__()
        self.mri_dir = mri_dir
        self.original_modal = original_modal
        self.target_modal = target_modal
        self.original_mri = [x + f"_{original_modal}.nii" for x in os.listdir(mri_dir)]
        self.target_mri = [x + f"_{target_modal}.nii" for x in os.listdir(mri_dir)]
        self.transform = transforms.Compose([transforms.Resize((256, 256)),
                                             transforms.Normalize(mean=(0.5), 
                                                std=(0.5)) # Normalization : -1 ~ 1 range
                                            ])
        self.len = len(self.original_mri)
    
    def __getitem__(self, index):
        original_images = nib.load(os.path.join(os.path.join(os.path.join(self.mri_dir, os.listdir(self.mri_dir)[index]), self.original_mri[index])))
        target_images = nib.load(os.path.join(os.path.join(os.path.join(self.mri_dir, os.listdir(self.mri_dir)[index]), self.target_mri[index])))
        original_images = original_images.get_fdata()
        target_images = target_images.get_fdata()

        #unsqueeze the image
        original_images = np.expand_dims(original_images, axis=0)
        target_images = np.expand_dims(target_images, axis=0)

        #covert to tensor
        original_images = torch.tensor(original_images, dtype=torch.float32)
        target_images = torch.tensor(target_images, dtype=torch.float32)
        
        return original_images, target_images
    
    def __len__(self):
        return self.len

# normailizes
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Conv -> Batchnorm -> Activate function Layer
def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True, activation='relu'):
    layers = []
    
    # Conv layer
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    
    # Batch Normalization
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    
    # Activation
    if activation == 'lrelu':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'none':
        pass
    
    return nn.Sequential(*layers)

# Deconv -> BatchNorm -> Activate function Layer
def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True, activation='lrelu'):
    layers = []
    
    # Deconv.
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    
    # Batchnorm
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    
    # Activation
    if activation == 'lrelu':
        layers.append(nn.LeakyReLU(0.2))
    elif activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'none':
        pass
                
    return nn.Sequential(*layers)

class Generator(nn.Module):
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        # Unet encoder
        self.conv1 = conv(1, 64, 4, bn=False, activation='lrelu') # (B, 64, 128, 128)
        self.conv2 = conv(64, 128, 4, activation='lrelu') # (B, 128, 64, 64)
        self.conv3 = conv(128, 256, 4, activation='lrelu') # (B, 256, 32, 32)
        self.conv4 = conv(256, 512, 4, activation='lrelu') # (B, 512, 16, 16)
        self.conv5 = conv(512, 512, 4, activation='lrelu') # (B, 512, 8, 8)
        self.conv6 = conv(512, 512, 4, activation='lrelu') # (B, 512, 4, 4)
        self.conv7 = conv(512, 512, 4, activation='lrelu') # (B, 512, 2, 2)
        self.conv8 = conv(512, 512, 4, bn=False, activation='relu') # (B, 512, 1, 1)

        # Unet decoder
        self.deconv1 = deconv(512, 512, 4, activation='relu') # (B, 512, 2, 2)
        self.deconv2 = deconv(1024, 512, 4, activation='relu') # (B, 512, 4, 4)
        self.deconv3 = deconv(1024, 512, 4, activation='relu') # (B, 512, 8, 8) # Hint : U-Net에서는 Encoder에서 넘어온 Feature를 Concat합니다! (Channel이 2배)
        self.deconv4 = deconv(1024, 512, 4, activation='relu') # (B, 512, 16, 16)
        self.deconv5 = deconv(1024, 256, 4, activation='relu') # (B, 256, 32, 32)
        self.deconv6 = deconv(512, 128, 4, activation='relu') # (B, 128, 64, 64)
        self.deconv7 = deconv(256, 64, 4, activation='relu') # (B, 64, 128, 128)
        self.deconv8 = deconv(128, 1, 4, activation='tanh') # (B, 1, 256, 256)

    # forward method
    def forward(self, input):
        # Unet encoder
        e1 = self.conv1(input)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        e5 = self.conv5(e4)
        e6 = self.conv6(e5)
        e7 = self.conv7(e6)
        e8 = self.conv8(e7)
                              
        # Unet decoder
        d1 = F.dropout(self.deconv1(e8), 0.5, training=True)
        d2 = F.dropout(self.deconv2(torch.cat([d1, e7], 1)), 0.5, training=True)
        d3 = F.dropout(self.deconv3(torch.cat([d2, e6], 1)), 0.5, training=True)
        d4 = self.deconv4(torch.cat([d3, e5], 1))
        d5 = self.deconv5(torch.cat([d4, e4], 1))
        d6 = self.deconv6(torch.cat([d5, e3], 1))
        d7 = self.deconv7(torch.cat([d6, e2], 1))
        output = self.deconv8(torch.cat([d7, e1], 1))
        
        return output

class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = conv(2, 64, 4, bn=False, activation='lrelu')
        self.conv2 = conv(64, 128, 4, activation='lrelu')
        self.conv3 = conv(128, 256, 4, activation='lrelu')
        self.conv4 = conv(256, 512, 4, 1, 1, activation='lrelu')
        self.conv5 = conv(512, 1, 4, 1, 1, activation='none')

    # forward method
    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        return out

if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    modal = ["flair", "t1", "t1ce", "t2"]

    train_dataset = BrainDataset("/data/brain/BraTS2020/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData", "t1", "t1ce")
    train_loader = DataLoader(dataset=train_dataset, num_workers=4, pin_memory=True, batch_size=200, shuffle=True) # Shuffle

    # Models
    G = Generator().cuda()
    D = Discriminator().cuda()

    criterionL1 = nn.L1Loss().cuda()
    criterionMSE = nn.MSELoss().cuda()

    # Setup optimizer
    g_optimizer = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Train
    for epoch in tqdm(range(100)):
        #use tqdm for batch process
        for i, (original_images, target_images) in enumerate(train_loader):
            #get rid of the first dimension
            # forward
            for slice in range(original_images.shape[4]):
                real_a = original_images[:, :, :, :, slice]
                real_b = target_images[:, :, :, :, slice]
                real_a = train_dataset.transform(real_a)
                real_b = train_dataset.transform(real_b)
                real_a = real_a.to(torch.float32).cuda()
                real_b = real_b.to(torch.float32).cuda()
            
                fake_b = G(real_a) # Generate the target image
                
                #============= Train the discriminator =============#
                # train with fake
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = D.forward(fake_ab.detach())
                fake_label = torch.zeros_like(pred_fake).cuda()
                loss_d_fake = criterionMSE(pred_fake, fake_label)

                # train with real
                real_ab = torch.cat((real_a, real_b), 1)
                pred_real = D.forward(real_ab)
                real_label = torch.ones_like(pred_fake).cuda()
                loss_d_real = criterionMSE(pred_real, real_label)
                
                # Combined D loss
                loss_d = (loss_d_fake + loss_d_real) * 0.5
                
                # Backprop + Optimize
                D.zero_grad()
                loss_d.backward()
                d_optimizer.step()

                #=============== Train the generator ===============#
                # First, G(A) should fake the discriminator
                fake_ab = torch.cat((real_a, fake_b), 1)
                pred_fake = D.forward(fake_ab)
                loss_g_gan = criterionMSE(pred_fake, real_label)

                # Second, G(A) = B
                loss_g_l1 = criterionL1(fake_b, real_b) * 10
                
                loss_g = loss_g_gan + loss_g_l1
                
                # Backprop + Optimize
                G.zero_grad()
                D.zero_grad()
                loss_g.backward()
                g_optimizer.step()
            
            if i == len(train_loader) - 1:
                print('======================================================================================================')
                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f'
                        % (epoch, 100, i, len(train_loader), loss_d.item(), loss_g.item()))
                print('======================================================================================================')

    # Save the model checkpoints
    torch.save(G.state_dict(), '/data/model/pix2pix/G.pth')
    torch.save(D.state_dict(), '/data/model/pix2pix/D.pth')