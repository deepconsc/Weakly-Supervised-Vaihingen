import cv2
import numpy as np 
import torch
from torch import nn
from torch.nn import functional as F
import albumentations as A
from torch.utils.data import Dataset
from random import randint 
import tqdm



"""
Data Mining Class.

Online augmentation with augmentor will be done. We discard any cropping techniques, as long as some class objects appear to be really small in 
mask images. Or we'd be confusing network.

HR image to tiles conversion is done separately beforehand.
"""

class DataGenerator(Dataset):
    def __init__(self, dataclass, train=True):
        self.images = dataclass[0] 
        self.labels = dataclass[1]
        self.train = train
        self.augmentor = A.Compose([    # Augmentor
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.OneOf([
            A.ElasticTransform(p=0.4, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=0.5)])
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):        
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.train:
            augmented_image = self.augmentor(image=image)['image']
        image = torch.tensor(cv2.resize(augmented_image, (200,200)), dtype=torch.float).permute(2,0,1)
        label = torch.tensor(label, dtype=torch.float)


        return [image, label]
    
                    