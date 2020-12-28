import numpy as np 
import cv2 
import torch.nn.functional as F 

"""
Class Activation Map generator function. 
Basically taken and modify from: https://gist.github.com/Arif159357/368de6364b0cd472897ae6665614c718#file-cam-py

We're iterating over batches and calculate class activation map over weights for each class.
"""

def generate_cam(feature_conv, weight):
    size_upsample = (200, 200)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for x in range(bz):
        local_cam = []
        for idx in range(5):
            beforeDot =  feature_conv[x].reshape((nc, h*w))
            cam = np.matmul(weight[idx], beforeDot)
            cam = cam.reshape(h, w)
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            local_cam.append(cv2.resize(cam_img, size_upsample))
        output_cam.append(local_cam)
    return output_cam