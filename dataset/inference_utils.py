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

def tilegenerator(image_paths):

    """
    Inputs 

    First input: Str [Path of images] 
    
    
    ```

    Outputs

    Output: List [idx_0 : np.array of images] - For inference

    """

    mask_paths = glob.glob('{image_paths}/*')
    tiles = []

    for e, img in tqdm(enumerate(image_paths)):
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

        # Calculating number of horizontal and vertical windows
        stride_hrz = img.shape[0]//200
        stride_vrt = img.shape[1]//200

        for y in range(stride_hrz+1):
            y0, y1 = (y*200, (y+1)*200) if y != stride_hrz else (y*200,img.shape[0])    # Move window horizontally, unless it's side part - then we'll use width as y1.

            for x in range(stride_vrt+1):
                x0, x1 = (x*200, (x+1)*200) if x != stride_vrt else (x*200,img.shape[1])    # Move window vertically, unless it's side part - then we'll use height as x1.
                img_processed = img[y0:y1, x0:x1,:]

                if img_processed.shape != (200, 200, 3):
                    img_processed = cv2.resize(img_processed, (200,200))
                    
                tiles.append(img_processed)


    return tiles


def imagegenerator(tiles):
    
    """
    Working on this yet.

    
    Inputs 

    First input: List [List of tiles] 
    
    
    ```

    Outputs

    Output: np.array [image] - For inference

    """

    
    tiles = []

    for e, img in tqdm(enumerate(image_paths)):
        img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

        # Calculating number of horizontal and vertical windows
        stride_hrz = img.shape[0]//200
        stride_vrt = img.shape[1]//200

        for y in range(stride_hrz+1):
            y0, y1 = (y*200, (y+1)*200) if y != stride_hrz else (y*200,img.shape[0])    # Move window horizontally, unless it's side part - then we'll use width as y1.

            for x in range(stride_vrt+1):
                x0, x1 = (x*200, (x+1)*200) if x != stride_vrt else (x*200,img.shape[1])    # Move window vertically, unless it's side part - then we'll use height as x1.
                img_processed = img[y0:y1, x0:x1,:]

                if img_processed.shape != (200, 200, 3):
                    img_processed = cv2.resize(img_processed, (200,200))
                    
                tiles.append(img_processed)


    return tiles