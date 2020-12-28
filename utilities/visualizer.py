import numpy as np
import glob
import matplotlib.pyplot as plt 

batch, epoch = 4, 7
images = []

pred, target = np.load(f'results/{epoch}_gen_image_{batch}.npy'), np.load(f'results/{epoch}_target_{batch}.npy')

for x in range(5):
    images.append(pred[:,:,x])
for x in range(5):
    images.append(target[:,:,x])
    
plt.figure(figsize=(15,15)) # specifying the overall grid size

for i in range(10):
    plt.subplot(5,5,i+1)    # the number of images in the grid is 5*5 (25)
    plt.imshow(images[i])

    
plt.show()
