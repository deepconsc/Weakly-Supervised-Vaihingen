import torch
from dataset.dataloader import DatasetFromFolder
from models.u2net import U2NET
from utils.iou import iou_calculation as iou
import argparse
from tqdm import tqdm 

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datadir', required=True, help='Path to validation dataset')
parser.add_argument('-c', '--checkpoint', required=True, help='Pretrained checkpoint')
params = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = U2NET()
model.load_state_dict(torch.load(args.checkpoint, map_location=device)['model_g'])
model.eval().to(device)


valdata = DatasetFromFolder(folder=args.datadir)
val_data_loader = torch.utils.data.DataLoader(dataset=valdata,
                                                batch_size=8,
                                                shuffle=False)

iou_stats = torch.zeros(5)
for i, (input, target) in tqdm(enumerate(val_data_loader)):
    
        pred = model(input.to(device))
        calculated_iou = iou(pred.detach().cpu(), target)
        iou_stats += calculated_iou

print(f'IoU calculation has been finished.')
print(f'Mean IoU: {torch.mean(iou_stats/i)}')
print(f'Classwise IoU: ')
for x in range(5):
    print(f'{x} - {iou_stats[x]/i}')
