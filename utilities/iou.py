import torch 


def iou_calculation(outputs, labels):
    
    intersection = (outputs & labels).float().sum((1, 2))  
    union = (outputs | labels).float().sum((1, 2)) 
    
    iou = (intersection + 1e-6) / (union + 1e-6)  
        
    return iou
