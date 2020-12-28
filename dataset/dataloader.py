import torch.utils.data as data
import os
import random
import numpy as np
import glob 
import torch
import torch.nn.functional as F
import cv2


class DatasetFromFolder(data.Dataset):
    def __init__(self, folder='train'):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = glob.glob(f'{folder}/*image*')
        self.direction = direction
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr
        self.cls_mapping = {0:[255, 255, 255], 1:[  0,   0, 255], 2:[  0, 255, 255], 3:[  0, 255,   0], 4:[255, 255,   0]}
    
    def __getitem__(self, index):
        # Load Image
        imgname = self.image_filenames[index]
        maskname = imgname.replace('image', 'mask')
        
        input = np.load(imgname)
        target = np.load(maskname)
        
        zero_mask = np.zeros((200,200,5))
        for key, value in self.cls_mapping.items():     # Iterating through class map to detect colors
            probs = cv2.inRange(target, np.array(value)-1, np.array(value)+1)
            if 255 in probs:   
                zero_mask[:,:,key] = probs/255
        
        input = torch.tensor((input - [0.5, 0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5, 0.5], dtype=torch.float).permute(2,0,1).unsqueeze(0)
        target = torch.tensor(zero_mask, dtype=torch.float).permute(2,0,1).unsqueeze(0)


        input = F.interpolate(input, (256,256)).squeeze(0)
        target = F.interpolate(target, (256,256)).squeeze(0)
        return input, target

    def __len__(self):
        return len(self.image_filenames)
