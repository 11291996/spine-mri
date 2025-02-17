import cv2
import torch
from os import listdir
from os.path import join
import torchvision.transforms as transforms
from torch.utils.data import Dataset

transform = transforms.Compose([   
    transforms.Normalize([0.5], [0.5]), 
    transforms.Resize((256,256))
])

class MRIDataset(Dataset):
    def __init__(self, path2img, original_modal, target_modal):
        super().__init__()
        self.path2a = join(path2img, original_modal)
        self.path2b = join(path2img, target_modal)
        self.img_filenames = [x for x in listdir(self.path2a)]

    def __getitem__(self, index):
        a = cv2.imread(join(self.path2a, self.img_filenames[index]), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(join(self.path2b, self.img_filenames[index]), cv2.IMREAD_UNCHANGED)

        a = torch.tensor(((a - a.min()) / (a.max() - a.min())), dtype=torch.float32)
        b = torch.tensor(((b - b.min()) / (b.max() - b.min())), dtype=torch.float32)

        a = a.unsqueeze(0)
        b = b.unsqueeze(0)

        a = transform(a)
        b = transform(b)

        return a, b

    def __len__(self):
        return len(self.img_filenames)