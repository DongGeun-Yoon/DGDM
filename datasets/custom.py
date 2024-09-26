import cv2
import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from Register import Registers
from PIL import Image

from datasets.mmnist import MovingMNIST
from datasets.typhoon import TLoader

@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.to_normal = dataset_config.to_normal

        self.is_train = True if stage=='train' else False
        if dataset_config.dataset == "MMNIST":
            self.images = MovingMNIST(root=dataset_config.dataset_path, is_train=self.is_train,)
        elif dataset_config.dataset == "typhoon":
            path = dataset_config.dataset_path \
                if self.is_train else dataset_config.valset_path
            self.images = TLoader(is_train=self.is_train, path=path)
        else:
            raise NotImplementedError
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        images = self.images[i]
        
        if not self.image_size[0] == images[0].size(2) == images[0].size(3):
            images_r1 = F.interpolate(images[0], size=self.image_size)
            images_r2 = F.interpolate(images[1], size=self.image_size)
            images = (images_r1, images_r2)
            
        return images[1], images[0]