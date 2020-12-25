import cv2
import torch
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import base64
from dataset.patches import patch, recover
from utils.pcam import generate_cam
from models.resnet import ResNet50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init():
    weights = glob.glob('*.pth')
    if 'resnet50.pth' not in weights:
        run('wget https://dummy.com -O resnet50.pth', shell=True)
    return True

def model_init():
    model = ResNet50()
    model.load_state_dict(torch.load('resnet50', map_location=device)['model'])
    model.eval().to(device)

    return model





def base64_to_image(imgstring):
    nparr = np.fromstring(base64.b64decode(imgstring), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape, tiles = patch(image)
    tensors = [torch.tensors(x, dtype=torch.float).permute(2,0,1).unsqueeze(0) for x in tiles]
    return tensors


app = FastAPI()
weights_init()
model  = model_init()

class JsonData(BaseModel):
    image: str



@app.post("/predict")
def mask_predictor(data: JsonData):
    image_array = base64_to_image(data.image)
    predicted_masks = []
    preprocessed_images = preprocess(image_array)
    for chunk in preprocessed_images:
        output, registered, weights = model(chunk.to(device))
        maps = generate_cam(registered, weights)[0]
        
    """
    Need to code visualization tool, it's same as we need in inference.py
    """
    
    retval, buffer = cv2.imencode('.png', dummy)
    buffer = base64.b64encode(buffer).decode("utf-8")
    return {'Prediction' : buffer}
    

