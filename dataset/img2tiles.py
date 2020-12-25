import cv2
import numpy as np 
from tqdm import tqdm
import glob 
from dataset.patches import patch

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
TOTAL_AREA = 200 * 200 

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

    mask_paths = glob.glob('gts_for_participants/*')[:num_images]
    image_paths = [x.replace('gts_for_participants/', 'top/') for x in mask_paths]

    total_images = []
    labels = []


    for e, img in tqdm(enumerate(image_paths)):
        image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        mask = cv2.cvtColor(cv2.imread(mask_paths[e]), cv2.COLOR_BGR2RGB)

        _shape, tiles = patch(image)
        shape_, masks = patch(mask)
        assert(_shape == shape_)

        for x in range(len(tiles)):
            zero_labels = np.zeros(5).astype(np.float32)  # Zero-array for multi-class labels.

            for key, value in CLS_MAPPING.items():     # Iterating through class map to detect colors
                probs = cv2.inRange(masks[x], np.array(value)-1, np.array(value)+1)
                if 255 in probs:      # This is a horrible workaround to detect color in tile. Needed to hardcode yet.
                    area = np.count_nonzero(probs == 255)
                    zero_labels[key] = area / TOTAL_AREA    # Let's calculate label area percentage for soft labeling

            labels.append(zero_labels)
            [total_images.append(x) for x in tiles]
    split_idx = int(len(tiles)*train_val_ratio)
    trainloader, valloader = (tiles[:split_idx], labels[:split_idx]) , (tiles[split_idx:], labels[split_idx:])

    return trainloader, valloader