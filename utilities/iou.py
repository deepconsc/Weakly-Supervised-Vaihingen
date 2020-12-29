import torch 
from sklearn.metrics import jaccard_similarity_score as jsc

def jaccard(pred, target):
    jac_classwise = []

    for x in range(pred.shape[0]):
        jac_classwise.append(jsc(pred[x].reshape(-1), target[x].reshape(-1)))
    return torch.tensor(jac_classwise)

def iou_calculation(pred, target):
    
    intersection = (pred & target).float().sum((1, 2))  
    union = (pred | target).float().sum((1, 2)) 
    
    iou = (intersection + 1e-6) / (union + 1e-6)  
        
    return iou