import time
import copy
from torch.nn import functional as F 
import torch 
import torch.nn as nn 
import numpy as np 
from collections import defaultdict
from tqdm import tqdm
from utils.pcam import generate_cam


def calc_loss(pred, target, features_conv, weights, metrics, bce_weight=0.5):
    """
    We calculate bce with logits loss for linear layer output over soft target labels.
    Additionally, treating class activation map area with >=90% probability as mask would be
    helpful for the model to learn. 

    229 in Line 23 is around 90% of 255, which is max pixel intensity in map. 40000 just stands for 200x200 image area.

    """
    bce = F.binary_cross_entropy(pred.cpu(), target.clone().cpu())
    
    area_ratios = []
    images = generate_cam(features_conv, weights)
    for img in images:
        area = np.count_nonzero(img>=229) / (40000)
        area_ratios.append(area)
    
    area_ratios = torch.tensor(area_ratios, requires_grad=True)
    area_loss = F.l1_loss(area_ratios.cpu(), target.cpu())

    loss = bce * bce_weight + area_loss * (1 - bce_weight)
    
    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['area'] += area_ratios.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
    return loss

def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))    

def train_model(model, optimizer, scheduler, num_epochs=25, dataloaders=None, device=None):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            iter = 0
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output, registered, weights = model(inputs)
                    loss = calc_loss(output, labels, registered, weights, metrics)/16  # Average loss by grad acc times

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if iter % 16 == 0:  
                            optimizer.step()    # if it's 16th iter, use the accumulated gradients

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'epoch': epoch+1,
                }, f'checkpoint_{epoch+1}.pth')

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
