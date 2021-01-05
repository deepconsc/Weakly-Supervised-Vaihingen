import cv2
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from dataset.dataloader import DataGenerator
import logging 
import torch.optim as optim
from torch.optim import lr_scheduler
from models.u2net import U2NET

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



import time
import copy
from torch.nn import functional as F 
import numpy as np 
from collections import defaultdict
from tqdm import tqdm


bce = nn.BCEWithLogitsLoss()
mse = nn.MSELoss()


def calc_loss(pred, target, metrics, bce_weight=0.5):

    bce_loss = bce(pred, target)
    mse_loss = mse(pred, target)

    loss = bce_loss * bce_weight + mse_loss * (1 - bce_weight)
    
    metrics['bce'] += bce_loss.data.cpu().numpy() * target.size(0)
    metrics['mse'] += mse_loss.data.cpu().numpy() * target.size(0)
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
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)             

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    output = model(inputs, classify=True)
                    loss = calc_loss(output, labels, metrics)# Average loss by grad acc times

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()    

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




if __name__ == '__main__':

    # Train data
    train_data = DatasetFromFolder(folder='train', pretraining=True)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                    batch_size=16,
                                                    shuffle=True, num_workers=8)

    # Test data
    test_data = DatasetFromFolder(folder='val', pretraining=True)
    test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                batch_size=4,
                                                shuffle=False, num_workers=8)

    test_input, test_target = test_data_loader.__iter__().__next__()




    dataset = {
        'train':train_data_loader,
        'val': test_data_loader
    }


    model = U2NET().cuda()

    for name, child in model.named_children():
        if name not in ['stage1', 'pool12', 'stage2', 'pool23', 'stage3', 'pool34', 'stage4', 'pool45', 'stage5', 'pool56', 'stage6','last_conv', 'fc']:
            for param in child.parameters():
                param.requires_grad = False




    optimizer_ft = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.5)             
    model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=200, dataloaders=dataset, device=device)




