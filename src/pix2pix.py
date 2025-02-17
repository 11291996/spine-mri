import os
from os import listdir
from os.path import join
import random
import matplotlib.pyplot as plt
import time
import torch
import numpy as np
import torch.nn as nn
import argparse

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]

        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels)),

        layers.append(nn.LeakyReLU(0.2))

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.down = nn.Sequential(*layers)

    def forward(self, x):
        x = self.down(x)
        return x

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels,4,2,1,bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        ]

        if dropout:
            layers.append(nn.Dropout(dropout))

        self.up = nn.Sequential(*layers)

    def forward(self,x,skip):
        x = self.up(x)
        x = torch.cat((x,skip),1)
        return x

class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64,128)                 
        self.down3 = UNetDown(128,256)               
        self.down4 = UNetDown(256,512,dropout=0.5) 
        self.down5 = UNetDown(512,512,dropout=0.5)      
        self.down6 = UNetDown(512,512,dropout=0.5)             
        self.down7 = UNetDown(512,512,dropout=0.5)              
        self.down8 = UNetDown(512,512,normalize=False,dropout=0.5)

        self.up1 = UNetUp(512,512,dropout=0.5)
        self.up2 = UNetUp(1024,512,dropout=0.5)
        self.up3 = UNetUp(1024,512,dropout=0.5)
        self.up4 = UNetUp(1024,512,dropout=0.5)
        self.up5 = UNetUp(1024,256)
        self.up6 = UNetUp(512,128)
        self.up7 = UNetUp(256,64)
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(128,out_channels,4,stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8,d7)
        u2 = self.up2(u1,d6)
        u3 = self.up3(u2,d5)
        u4 = self.up4(u3,d4)
        u5 = self.up5(u4,d3)
        u6 = self.up6(u5,d2)
        u7 = self.up7(u6,d1)
        u8 = self.up8(u7)

        return u8

class Dis_block(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()

        layers = [nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
    
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        x = self.block(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.stage_1 = Dis_block(in_channels*2,64,normalize=False)
        self.stage_2 = Dis_block(64,128)
        self.stage_3 = Dis_block(128,256)
        self.stage_4 = Dis_block(256,512)

        self.patch = nn.Conv2d(512,1,3,padding=1) # 16x16 패치 생성

    def forward(self,a,b):
        x = torch.cat((a,b),1)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.patch(x)
        x = torch.sigmoid(x)
        return x

def initialize_weights(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from .utils import MRIDataset
    from torch import optim

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=str, default="1")
    parser.add_argument("--data_dir", type=str, default="/data/datasets/spine/gtu/train")
    parser.add_argument("--original_modal", type=str, default="t1")
    parser.add_argument("--target_modal", type=str, default="t2")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="/data/model")
    args = parser.parse_args()

    model_type = "pix2pix"

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = MRIDataset(args.data_dir, args.original_modal, args.target_modal)
    train_loader = DataLoader(dataset=train_dataset, num_workers=4, pin_memory=True, batch_size=args.batch_size, shuffle=True)

    model_gen = GeneratorUNet()
    model_dis = Discriminator()

    model_gen.apply(initialize_weights)
    model_dis.apply(initialize_weights)

    loss_func_gan = nn.BCELoss()
    loss_func_pix = nn.L1Loss()

    lambda_pixel = 100

    patch = (1,256//2**4,256//2**4)

    lr = args.lr
    beta1 = args.beta1
    beta2 = args.beta2

    opt_dis = optim.Adam(model_dis.parameters(),lr = lr, betas=(beta1, beta2))
    opt_gen = optim.Adam(model_gen.parameters(),lr = lr, betas=(beta1, beta2))

    model_gen.to(device)
    model_dis.to(device)

    model_gen.train()
    model_dis.train()

    num_epochs = args.num_epochs
    start_time = time.time()

    loss_hist = {'gen':[],
                'dis':[]}

    for epoch in range(num_epochs):
        for a, b in train_loader:

            ba_si = a.size(0)

            # real image
            real_a = a.to(device)
            real_b = b.to(device)

            # patch label
            real_label = torch.ones(ba_si, *patch, requires_grad=False).to(device)
            fake_label = torch.zeros(ba_si, *patch, requires_grad=False).to(device)

            # generator
            model_gen.zero_grad()

            fake_b = model_gen(real_a) 
            out_dis = model_dis(fake_b, real_b) 

            gen_loss = loss_func_gan(out_dis, real_label)
            pixel_loss = loss_func_pix(fake_b, real_b)

            g_loss = gen_loss + lambda_pixel * pixel_loss
            g_loss.backward()
            opt_gen.step()

            # discriminator
            model_dis.zero_grad()

            out_dis = model_dis(real_b, real_a) 
            real_loss = loss_func_gan(out_dis,real_label)
        
            out_dis = model_dis(fake_b.detach(), real_a) 
            fake_loss = loss_func_gan(out_dis,fake_label)

            d_loss = (real_loss + fake_loss) / 2.
            d_loss.backward()
            opt_dis.step()

            loss_hist['gen'].append(g_loss.item())
            loss_hist['dis'].append(d_loss.item())

            print('Epoch: %.0f, G_Loss: %.6f, D_Loss: %.6f, time: %.2f min' %(epoch, g_loss.item(), d_loss.item(), (time.time()-start_time)/60))

    subject_name = args.data_dir.split('/')[-3]
    dataset_name = args.data_dir.split('/')[-2]

    if not os.path.exists(os.path.join(args.save_dir, dataset_name)):
        os.makedirs(os.path.join(args.save_dir, model_type, subject_name, dataset_name))

    # save model
    torch.save(model_gen.state_dict(), os.path.join(args.save_dir, model_type, subject_name, dataset_name, f'G_{args.original_modal}_{args.target_modal}.pth'))
    torch.save(model_dis.state_dict(), os.path.join(args.save_dir, model_type, subject_name, dataset_name, f'D_{args.original_modal}_{args.target_modal}.pth'))

