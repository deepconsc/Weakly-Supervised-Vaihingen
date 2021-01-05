import torch.utils.data as data
import os
import random
import numpy as np
import glob 
import torch
import torch.nn.functional as F
import cv2


class DatasetFromFolder(data.Dataset):
    def __init__(self, folder='train', pretraining=False):
        super(DatasetFromFolder, self).__init__()
        self.pretraining = pretraining
        if not self.pretraining and folder == 'train':
            self.image_filenames = glob.glob(f'train/*image*')[:6000]
        if not self.pretraining and folder == 'val':
            self.image_filenames = glob.glob(f'val/*image*')
        if self.pretraining:
            self.image_filenames = glob.glob(f'{folder}/*image*')


        self.cls_mapping = {0:[255, 255, 255], 1:[  0,   0, 255], 2:[  0, 255, 255], 3:[  0, 255,   0], 4:[255, 255,   0]}
        
    def __getitem__(self, index):
        # Load Image
        imgname = self.image_filenames[index]
        maskname = imgname.replace('image', 'mask')
        
        input = np.load(imgname)
        target = np.load(maskname)

        if self.pretraining:
            zero_labels = np.zeros(5).astype(np.float32)  

            for key, value in self.cls_mapping.items():    
                probs = cv2.inRange(target, np.array(value)-1, np.array(value)+1)
                if 255 in probs:      
                    zero_labels[key] = 1.0  
            input = torch.tensor((input - [0.5, 0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5, 0.5], dtype=torch.float).permute(2,0,1).unsqueeze(0)
            input = F.interpolate(input, (256,256)).squeeze(0)
            target = torch.tensor(zero_labels, dtype=torch.float)         

        else: 
            zero_mask = np.zeros((200,200,5))
            for key, value in self.cls_mapping.items():     
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
