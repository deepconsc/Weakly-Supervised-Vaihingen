import cv2
import numpy as np 
from tqdm import tqdm
import glob 


"""
Tile Generator function.
We'd be loading images with labels in memory during initialization. As long as we need to calculate colorspaces in tiles to generate class labels, 
we don't need that happening during __getitem__.  


Color mapping would be the following:

0. Impervious surfaces - WHITE
1. Building - BLUE
2. Low vegetation - TURQUOISE
3. Tree - GREEN
4. Car - YELLOW

"""

CLS_MAPPING = {0:[255, 255, 255], 1:[  0,   0, 255], 2:[  0, 255, 255], 3:[  0, 255,   0], 4:[255, 255,   0]}

def tilegenerator(image_paths, num_images, train_val_ratio):

    """
    Inputs 

    First input: Str [Path of images] 
    Second input: Float [train / (total dataset)]
    
    
    ```

    Outputs

    First output: List [idx_0 : np.array of image,  idx_1 : np.array of soft labels] - For training
    Second output: List [idx_0 : np.array of image,  idx_1 : np.array of soft labels] - For validation

    """

    image_paths = glob.glob(f'{image_paths}/*')  # Retrieve specific amount of img paths
    mask_paths = [x.replace('{image_paths}/', 'gts_for_participants/') for x in image_paths] # Convert to mask paths

    tiles = []
    generated_labels = []


    for e, img in tqdm(enumerate(image_paths)):
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        mask_image = cv2.cvtColor(cv2.imread(mask_paths[e]), cv2.COLOR_BGR2RGB)

        assert(mask_image.shape == img.shape)

        # Calculating number of horizontal and vertical windows
        stride_hrz = img.shape[0]//200
        stride_vrt = img.shape[1]//200

        for y in range(stride_hrz+1):
            y0, y1 = (y*200, (y+1)*200) if y != stride_hrz else (y*200,img.shape[0])    # Move window horizontally, unless it's side part - then we'll use width as y1.

            for x in range(stride_vrt+1):
                zero_labels = np.zeros(5).astype(np.float32)  # Zero-array for multi-class labels.
                x0, x1 = (x*200, (x+1)*200) if x != stride_vrt else (x*200,img.shape[1])    # Move window vertically, unless it's side part - then we'll use height as x1.
                mask = mask_image[y0:y1, x0:x1, :]
                img_processed = img[y0:y1, x0:x1,:]
                if mask.shape != (200, 200, 3):
                    mask = cv2.resize(mask, (200, 200))
                    img_processed = cv2.resize(img_processed, (200,200))
                    
                total_area = mask.shape[0] * mask.shape[1]

                for key, value in CLS_MAPPING.items():     # Iterating through class map to detect colors
                    probs = cv2.inRange(mask, np.array(value)-1, np.array(value)+1)
                    if 255 in probs:      # This is a horrible workaround to detect color in tile. Needed to hardcode yet.
                        area = np.count_nonzero(probs == 255)
                        zero_labels[key] = area / total_area    # Let's calculate label area percentage for soft labeling

                tiles.append(img_processed)
                generated_labels.append(zero_labels)

    split_idx = int(len(tiles)*train_val_ratio)
    trainloader, valloader = (tiles[:split_idx], generated_labels[:split_idx]) , (tiles[split_idx:], generated_labels[split_idx:])

    return trainloader, valloader