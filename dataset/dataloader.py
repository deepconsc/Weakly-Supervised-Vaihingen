import torch.utils.data as data
import os
import random
import numpy as np
import glob 
import torch
import torch.nn.functional as F
import cv2


class DatasetFromFolder(data.Dataset):
    '''
    Dataset class.
    Contains dataloaders for both - pretraining (classifier) and training (segmentation) approaches.
    As long as we're handling with 4 and 5 channel tensors, augmentation is a bit tricky. 

    I've implemented torch.: flipud, fliplr, rot90 augmentation methods to handle images as raw tensors.
    '''
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

            # aug_1 = random.randint(0,1) #fliplr
            # aug_2 = random.randint(0,1) #flipud
            # aug_3 = random.randint(0,1) #rot90
            # k = random.randint(0, 3)

            # if aug_1 == 1:
            #     input = torch.fliplr(input)
            #     target = torch.fliplr(target)
            # if aug_2 == 1:
            #     input = torch.flipud(input)
            #     target = torch.flipud(target)
            # if aug_3 == 1:
            #     input = torch.rot90(input, k, [-2, -1])
            #     target = torch.rot90(target, k, [-2,-1])



        return input, target

    def __len__(self):
        return len(self.image_filenames)
