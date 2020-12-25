import cv2
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from dataset.img2tiles import tilegenerator
from dataset.dataloader import DataGenerator
import logging 
from models.resnet import ResNet50
from utils.pcam import generate_cam
import yaml 
import torch.optim as optim
from torch.optim import lr_scheduler
from utils.train import train_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config = yaml.load(open('configs/config.yaml'), Loader=yaml.FullLoader)

train_set, val_set = tilegenerator(image_paths=config['train']['folder'], num_images=config['train']['num_images'], train_val_ratio=0.9)

image_datasets = {
    'train': train_set, 'val': val_set
}

batch_size = config['train']['batch_size']

dataloaders = {
    'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=config['train']['num_workers']),
    'val': DataLoader(val_set, batch_size=4, shuffle=True, num_workers=0)
}

dataset_sizes = {
    x: len(image_datasets[x]) for x in image_datasets.keys()
}

logging.info(dataset_sizes)

model = ResNet50().cuda()


optimizer_ft = optim.SGD(model.parameters(), lr=config['optimizer']['lr'], momentum=config['optimizer']['momentum'])
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=config['optimizer']['scheduler_steps'], gamma=config['optimizer']['gamma'])             
model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=config['train']['epochs'], dataloaders)