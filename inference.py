import cv2
import torch 
import torch.nn as nn 
from dataset.patches import patch, recover
from utils.pcam import generate_cam
from models.resnet import ResNet50
import argparse
import glob 



def preprocess(image_path):
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    shape, tiles = patch(image)
    tensors = [torch.tensors(x, dtype=torch.float).permute(2,0,1).unsqueeze(0) for x in tiles]
    return tensors

#def postproceess_maps(maps):

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, default=None, required=True,
                    help="path of the input files directory")
parser.add_argument('-c', '--checkpoint', type=str, required=True,
                    help="path of the checkpoint")
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = ResNet50()
if torch.cuda.is_available():
    model.load_state_dict(torch.load(args.checkpoint)['model'])
else:
    model.load_state_dict(torch.load(args.checkpoint, map_location='cpu')['model'])

model.eval().to(device)

for image_path in glob.glob(args.input):
    predicted_masks = []
    preprocessed_images = preprocess(image_path)
    for chunk in preprocessed_images:
        output, registered, weights = model(chunk.to(device))
        maps = generate_cam(registered, weights)[0]

        """
        Need to postprocess maps here to generate masks.
        """

