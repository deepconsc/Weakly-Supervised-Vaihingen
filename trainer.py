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
from utils import plot 
import argparse
import os
from logger import Logger
from torch import nn 
from u2net import U2NET

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='facades', help='input dataset')
parser.add_argument('--direction', required=False, default='BtoA', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=200, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0002, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()
print(params)


save_dir = 'results/'
model_dir = 'model/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

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
train_data = DatasetFromFolder('', subfolder='train', direction=params.direction, transform=None,
                               resize_scale=params.resize_scale, crop_size=params.crop_size, fliplr=params.fliplr)
train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=params.batch_size,
                                                shuffle=True)

# Test data
test_data = DatasetFromFolder('', subfolder='val', direction=params.direction, transform=None)
test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=params.batch_size,
                                               shuffle=False)
test_input, test_target = test_data_loader.__iter__().__next__()


G = U2NET()
D = Discriminator(9, params.ndf, 1)
G.cuda()
D.cuda()
D.normal_weight_init(mean=0.0, std=0.02)

# Set the logger
D_log_dir = save_dir + 'D_logs'
G_log_dir = save_dir + 'G_logs'
if not os.path.exists(D_log_dir):
    os.mkdir(D_log_dir)
D_logger = Logger(D_log_dir)

if not os.path.exists(G_log_dir):
    os.mkdir(G_log_dir)
G_logger = Logger(G_log_dir)

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
        gen_image, d1, d2, d3, d4, d5, d6 = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        fake_ = Variable(torch.zeros(D_fake_decision.size()).cuda())
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        # Back propagation
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        gen_image, d1, d2, d3, d4, d5, d6 = G(x_)
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

        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
              % (epoch+1, params.num_epochs, i+1, len(train_data_loader), D_loss.item(), G_loss.item()))

        # ============ TensorBoard logging ============#
        D_logger.scalar_summary('losses', D_loss.item(), step + 1)
        G_logger.scalar_summary('losses', G_loss.item(), step + 1)
        step += 1

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

    # avg loss values for plot
    D_avg_losses.append(D_avg_loss)
    G_avg_losses.append(G_avg_loss)
    
    # Show result for test image
    gen_image,d1, d2, d3, d4, d5, d6 = G(Variable(test_input.cuda()))
    gen_image = gen_image.cpu().data
    plot.plot_test_result(test_input, test_target, gen_image, epoch, save=True, save_dir=save_dir)

# Plot average losses
plot.plot_loss(D_avg_losses, G_avg_losses, params.num_epochs, save=True, save_dir=save_dir)

# Make gif
plot.make_gif(params.dataset, params.num_epochs, save_dir=save_dir)

# Save trained parameters of model
torch.save(G.state_dict(), model_dir + 'generator_param.pkl')
torch.save(D.state_dict(), model_dir + 'discriminator_param.pkl')
