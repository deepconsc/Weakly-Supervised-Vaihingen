import numpy as np 
import torch
import torch.nn as nn
from torchvision import models 
import torch.nn.functional as F


class ResNet50(nn.Module):
    """
    Resnet Class. We're going to use class activation map as auxiliary loss,
    so we'd be returning last conv map and fc layer weights for every iteration.
    """

    def __init__(self):
        super().__init__()
        
        self.base_model = models.resnet50(pretrained=True)
        
        self.base_layers = list(self.base_model.children())                
        
        self.conv = nn.Sequential(*self.base_layers[:-2])
        self.adaptivepool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features = 2048, out_features=5)


        
    def forward(self, x):
        x = self.conv(x)
        registered = x.clone().detach().cpu().numpy()
        x = self.adaptivepool(x)
        x = x.view(x.shape[0], 2048)
        x = self.fc(x)
        weights = np.squeeze(F.softmax(list(model.fc.parameters())[-2]).detach().cpu().numpy())
        return x, registered, weights
