import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
from PIL import Image, ImageFilter
import glob
import random

class TLoader(Dataset):
    """
    is_train - 0 train, 1 val
    """
    def __init__(self, is_train=0, path='crop_typhoon/', transform=None):
        self.is_train = is_train
        self.transform = transform
        self.path = glob.glob(f"{path}/*") * 50
        self.totenosr = transforms.ToTensor()

    def __len__(self):
        return len(self.path)

    def _make_sqe(self, path):
        if self.is_train:
            images = glob.glob(f"{path}/*/*.png")
            images = [i for i in images if 'ir105' in i]
            images = sorted(images)
            s_index = np.random.randint(len(images)-20)
        else:
            images = glob.glob(f"{path}/*/*.png")
            images = [i for i in images if 'ir105' in i]
            images = sorted(images)
            s_index = 0 
        return images[s_index:s_index+10], images[s_index+10:s_index+20]

    def _preprocessor(self, path, is_input=False):
        imgs = []
        for i in path:
            ir = np.array(Image.open(i).convert('L'))
            sw = np.array(Image.open(i.replace('ir105', 'sw038')).convert('L'))
            wv = np.array(Image.open(i.replace('ir105', 'wv063')).convert('L'))
            img = np.stack([ir, sw, wv], axis=0)
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)
        imgs = torch.from_numpy(imgs)
        imgs = imgs / 255.
        return imgs

    def __getitem__(self, idx):
        clip_path = self.path[idx]
        inputs, outputs = self._make_sqe(clip_path)
        
        inputs = self._preprocessor(inputs, True)
        outputs = self._preprocessor(outputs, True)
        
        inputs = torch.clamp(inputs, 0, 1)
        outputs = torch.clamp(outputs, 0, 1)

        inputs = inputs * 2 - 1
        outputs = outputs * 2 - 1
        return inputs, outputs    
    