import torch
from dataset.dataloader import DatasetFromFolder
from models.u2net import U2NET
from utilities.iou import metrics
import argparse
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datadir', required=True, help='Path to validation dataset')
parser.add_argument('-c', '--checkpoint', required=True, help='Pretrained checkpoint')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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




