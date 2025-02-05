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

class MendeleyDataset(data.Dataset):
    mri_dir = "/data/spine/mendeley-lumbar/01_MRI_Data"
    modal_dir = {
        "t1_sag" : "",
        "t1_tra" : "",
        "t2_sag" : "",
        "t2_tra" : "",
        "localizer" : ""
    }
    
    def __init__(self, axis: list, original_modal, target_modal):
        
        self.axis = axis
        self.original_modal = original_modal
        self.target_modal = target_modal
        
        self.original_mri = [x + f"_{original_modal}.nii" for x in os.listdir(mri_dir)]
        self.target_mri = [x + f"_{target_modal}.nii" for x in os.listdir(mri_dir)]
        os.path.join(os.path.join(os.path.join(self.mri_dir, os.listdir(self.mri_dir)[index]), self.original_mri[index]))
        
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
