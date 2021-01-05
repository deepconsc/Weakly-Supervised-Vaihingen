import torch
from dataset.patch_recover import patch as cutout
from dataset.patch_recover import recover
import cv2 
import numpy as np 
import logging 
from pydantic import BaseModel
from generate_data import process

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inference(img):
        shape = img.shape
        processed = []
        _, images = cutout(img)
        for x in tqdm.tqdm(range(len(images))):
                input = process(images[x])
                input = torch.tensor((input - [0.5, 0.5, 0.5, 0.5]) / [0.5, 0.5, 0.5, 0.5], dtype=torch.float).permute(2,0,1).unsqueeze(0)
                input = F.interpolate(input, (256,256)).squeeze(0).to(device)
                mask = model(input, classify=False).detach().cpu().numpy().permute(1,2,0)
                processed.append(mask)
        out = recover(shape, processed)
        return out



def b2i(imgstring):
    nparr = np.fromstring(base64.b64decode(imgstring), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def weights_init():
    weights = glob.glob('*.cpp')
    while True:
        if 'u2net_graph.cpp' not in weights:
                logging.info('Downloading model graph...')
                run('wget -q https://www.dropbox.com/s/eoclzmwcefhia0c/u2net_graph.cpp?dl=0 -O u2net_graph.cpp', shell=True)

        weights = glob.glob('*.pth')
        graphbytes = check_output('du u2net_graph.cpp', shell=True).decode('utf-8').split('\t')[0] == '173012'

        logging.info('Download finished.')
        logging.info('Checking for file sizes.')

        if 'u2net_graph.cpp' in weights and (not graphbytes):
                logging.warning('Model graph is corrupted, will retry.')
                run('wget -q https://www.dropbox.com/s/eoclzmwcefhia0c/u2net_graph.cpp?dl=0 -O u2net_graph.cpp', shell=True)
                sebytes = check_output('du u2net_graph.cpp', shell=True).decode('utf-8').split('\t')[0] == '173012'
                weights = glob.glob('*.pth')
        if 'u2net_graph.cpp' in weights and graphbytes:
                logging.info('Model checkpoints are valid.')
                break
    return model

weights_init()
model = torch.jit.load('u2net_graph.cpp').to(device).eval()


class JsonData(BaseModel):
    image: str



