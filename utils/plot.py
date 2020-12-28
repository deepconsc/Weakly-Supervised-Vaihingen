import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import numpy as np

# For logger
def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Plot losses
def plot_loss(d_losses, g_losses, num_epochs, save=False, save_dir='results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'Loss_values_epoch_{:d}'.format(num_epochs) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_test_result(input, target, gen_image, epoch, training=True, save=False, save_dir='results/', show=False, fig_size=(5, 5)):
    for x in range(8):
        np.save(f'results/{epoch+1}_input_{x}', input[x].squeeze(0).numpy().transpose(1,2,0))
        np.save(f'results/{epoch+1}_target_{x}', target[x].squeeze(0).numpy().transpose(1,2,0))
        np.save(f'results/{epoch+1}_gen_image_{x}', gen_image[x].squeeze(0).numpy().transpose(1,2,0))

  
