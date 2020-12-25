import cv2
import torch 
import torch.nn as nn 
from dataset.img2tiles import tilegenerator
from models.resnet import ResNet50
import argparse

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

