import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class SimSiamDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        if self.transform is not None:
            img1, img2 = self.transform(img)
            return img1, img2
        else:
            return img, img