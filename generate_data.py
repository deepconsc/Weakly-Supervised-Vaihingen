import cv2
import numpy as np
from math import ceil
from random import randint

import glob
import tqdm
import cv2
import torch 
import numpy as np 
from subprocess import run 
from dataset.patch_recover import patch as cutout

parser = argparse.ArgumentParser()
parser.add_argument('--gt', type=str, default='mosaic_gts', help='ground-truth images directory')
parser.add_argument('--images', type=str, default='top', help='images directory')
args = parser.parse_args()


localfiles = glob.glob('*')
if 'train' not in localfiles and 'val' not in localfiles:
    run('mkdir train val', shell=True)

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
    
    
        
def patch(img, mask, num_generated, count, train=True, custom=False):

    if train:
        path = 'train'
    else: 
        path = 'val'

    if not custom:
        max_start_width = img.shape[1] - 100
        max_start_height = img.shape[0] - 100
        
        for x in tqdm.tqdm(range(num_generated)):
            center = (randint(100, max_start_height), randint(100, max_start_width))
            temporary_img = process(img[center[0]-100:center[0]+100, center[1]-100:center[1]+100])
            temporary_mask = mask[center[0]-100:center[0]+100, center[1]-100:center[1]+100]
            
            if train:
                path = 'train'
            else: 
                path = 'val'
            np.save(f'{path}/image_{x}_{count}', temporary_img)
            np.save(f'{path}/mask_{x}_{count}', temporary_mask)

    if custom: 
        _, images = cutout(img)
        _, masks = cutout(mask)
        for x in tqdm.tqdm(range(len(images))):
            temporary_img = process(images[x])
            temporary_mask = masks[x]
            np.save(f'validation/image_{x}_{count}', temporary_img)
            np.save(f'validation/mask_{x}_{count}', temporary_mask)


        
        
def generator(num_gt, num_total, train, custom): 
    if custom:
        mask_paths = glob.glob(f'{args.gt}/*')[3:3+num_gt]
    else:
        mask_paths = glob.glob(f'{args.gt}/*')[:num_gt]
    image_paths = [x.replace(f'{args.gt}', f'{args.images}') for x in mask_paths]
    for a in range(len(image_paths)):
        image = cv2.cvtColor(cv2.imread(image_paths[a]), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_paths[a]), cv2.COLOR_BGR2RGB)
        patch(image, mask, num_total, a, train, custom)






midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").cuda().eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Generate train images. We generate 1000 tile per HR image, 
# 3000 will be used with pixel-level annotations, rest for classification
generator(num_gt=26, num_total=1000, train=True, custom=False)
generator(num_gt=3, num_total=300, train=False, custom=False)