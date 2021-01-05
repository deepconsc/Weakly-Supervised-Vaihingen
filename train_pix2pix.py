"""
Pix2Pix with U^2-Net Generator Trainer.
Code is adopted from: https://github.com/togheppi/pix2pix

It needs to be refactored yet.
"""
import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset.dataloader import DatasetFromFolder
from models.u2net import U2NET
from models.discriminator import Discriminator
from utilities import plot 
import argparse
import os, sys, glob
from torch import nn 
import logging
from utilities.iou import jaccard

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
parser.add_argument('--num_epochs', type=int, default=200, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()
print(params)


save_dir = 'results/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


bce_loss_ = nn.BCELoss(size_average=True)

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss_(d0,labels_v)
	loss1 = bce_loss_(d1,labels_v)
	loss2 = bce_loss_(d2,labels_v)
	loss3 = bce_loss_(d3,labels_v)
	loss4 = bce_loss_(d4,labels_v)
	loss5 = bce_loss_(d5,labels_v)
	loss6 = bce_loss_(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

	return loss



# Train data
train_data = DatasetFromFolder(folder='train')
train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=params.batch_size,
                                                shuffle=True)

# Test data
test_data = DatasetFromFolder(folder='val')
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=8,
                                               shuffle=False)

test_input, test_target = test_data_loader.__iter__().__next__()

chks = sorted(glob.glob('*.pth'), key=os.path.getmtime)

G = U2NET()
G.load_state_dict(torch.load(chks[-1])['model'])

for name, child in G.named_children():
    if name in ['stage1', 'pool12', 'stage2', 'pool23', 'stage3', 'pool34', 'stage4', 'pool45', 'stage5', 'pool56', 'stage6','last_conv', 'fc']:
        for param in child.parameters():
            param.requires_grad = False
G.cuda()
D = Discriminator(9, 64, 1)
D.cuda()
D.normal_weight_init(mean=0.0, std=0.02)


# Loss function
BCE_loss = torch.nn.BCELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2))
D_optimizer = torch.optim.Adam(D.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))

# Training GAN
D_avg_losses = []
G_avg_losses = []

step = 0
for epoch in range(params.num_epochs):
    D_losses = []
    G_losses = []

    # training
    for i, (input, target) in enumerate(train_data_loader):

        # input & target image data
        x_ = Variable(input.cuda())
        y_ = Variable(target.cuda())

        # Train discriminator with real data
        D_real_decision = D(x_, y_).squeeze()
        real_ = Variable(torch.ones(D_real_decision.size()).cuda())
        D_real_loss = BCE_loss(D_real_decision, real_)
        # Train discriminator with fake data
        gen_image, d1, d2, d3, d4, d5, d6 = G(x_, classify=False)
        D_fake_decision = D(x_, gen_image).squeeze()
        fake_ = Variable(torch.zeros(D_fake_decision.size()).cuda())
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        # Back propagation
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        gen_image, d1, d2, d3, d4, d5, d6 = G(x_, classify=False)
        auxiliary = muti_bce_loss_fusion(gen_image, d1, d2, d3, d4, d5, d6, y_)
        D_fake_decision = D(x_, gen_image).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, real_)

        # L1 loss
        l1_loss = params.lamb * L1_loss(gen_image, y_)

        # Back propagation
        G_loss = G_fake_loss + l1_loss + auxiliary
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss values
        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())
        
        if step % 50 == 0:
            print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                  % (epoch+1, params.num_epochs, i+1, len(train_data_loader), D_loss.item(), G_loss.item()))
            sys.stdout.flush()

        step += 1

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))
    if epoch+1 % 5 == 0:
        torch.save({
                    'model_g': G.state_dict(),
                    'model_d': D.state_dict(),
                }, f'model_epoch_{epoch}.pth')
        G.eval()
        iou_stats = torch.zeros(5)
        for i, (input, target) in enumerate(test_data_loader):
            
                pred, d1, d2, d3, d4, d5, d6 = G(input.cuda(), classify=False)
                pred = pred.detach().cpu().int()
                target = target.int()
                for x in range(pred.shape[0]):
                    calculated_iou = jaccard(pred[x],target[x])
                    iou_stats += calculated_iou/pred.shape[0]

        print(f'Mean IoU: {torch.mean(iou_stats/i)*100:.2f}')
        print(f'Classwise IoU: ')
        for x in range(5):
            print(f'{x} - {iou_stats[x]/i*100:.2f}')
        G.train()
    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)
    
    # Show result for test image
    gen_image,d1, d2, d3, d4, d5, d6 = G(Variable(test_input.cuda()), classify=False)
    gen_image = gen_image.cpu().data
    plot.plot_test_result(test_input, test_target, gen_image, epoch, save=True, save_dir=save_dir)
