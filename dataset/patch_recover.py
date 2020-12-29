import cv2
import numpy as np
from math import ceil

def patch(img):
    """
    Patching algorithm. 
    Takes image and recreates 200x200xC patches. It could be done better, but
    after torch.nn-s ridiculous fold/unfold and sklearn's memory error, this works smoothly.
    ```
    Outputs
    First output: Tuple [Image shape]
    Second output: List [Array of patch images] 
    """
    h_pad = ceil(img.shape[0]/200) 
    w_pad = ceil(img.shape[1]/200)
    zero_image = np.zeros((h_pad*200, w_pad*200, 3))
    zero_image[:img.shape[0], :img.shape[1], :] = img
    images = []
    for i in range(h_pad):
        for j in range(w_pad):
            if i == 0 and j != 0:
                crop = zero_image[i:i+200, j*200:(j+1)*200, :]
                images.append(crop)
            elif i != 0 and j == 0:
                crop = zero_image[i*200:(i+1)*200, j:j+200, :]
                images.append(crop)
            elif i == 0 and j == 0:
                crop = zero_image[i:i+200, j:j+200, :]
                images.append(crop)
            else:
                crop = zero_image[i*200:i*200+200, j*200:j*200+200, :]
                images.append(crop)
    return zero_image.shape, images


def recover(shape, chunks):

    """
    Recovering from patches. 
    Takes array of chunks (patches) and recreates image with sizes of 'shape' variable.
    ```
    Outputs
    Output: Array [Image]
    """

    h,w = shape[0]//200, shape[1]//200
    hor = []
    for i in range(0,h):
        im = cv2.hconcat([i for i in chunks[i*w:i*w+w]])
        hor.append(im)
    image = cv2.vconcat(hor)
    return image
