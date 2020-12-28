import cv2
import numpy as np
from math import ceil
from random import randint

import glob
import tqdm
import cv2
import torch 
import numpy as np 

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").cuda().eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

mask_paths = glob.glob('gts_for_participants/*')[:3]
image_paths = [x.replace('gts_for_participants/', 'top/') for x in mask_paths]
print(image_paths, mask_paths)
def process(img):
    input_batch = transform(img).cuda()
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        output = prediction.cpu().numpy()[:,:,np.newaxis]
        output /= (np.max(output)/255)
        four_ch = np.concatenate((img, output), axis=2)
    return four_ch
    
    
        
def patch(img, mask, num_generated, count):
    max_start_width = img.shape[1] - 100
    max_start_height = img.shape[0] - 100
    print(img.shape)
    
    for x in tqdm.tqdm(range(num_generated)):
        center = (randint(100, max_start_height), randint(100, max_start_width))
        temporary_img = process(img[center[0]-100:center[0]+100, center[1]-100:center[1]+100])
        temporary_mask = mask[center[0]-100:center[0]+100, center[1]-100:center[1]+100]
        
        np.save(f'train/image_{x}_{count}', temporary_img )
        np.save(f'train/mask_{x}_{count}', temporary_mask )
        
        
        
for a in range(len(image_masks)):
    image = cv2.cvtColor(cv2.imread(image_paths[a]), cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(cv2.imread(mask_paths[a]), cv2.COLOR_BGR2RGB)
    patch(image, mask, 3000, a)
