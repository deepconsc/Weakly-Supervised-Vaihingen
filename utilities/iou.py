import torch 
from sklearn.metrics import jaccard_similarity_score as jsc
import sklearn

def jaccard(pred, target):
    jac_classwise = []

    for x in range(pred.shape[0]):
        jac_classwise.append(jsc(pred[x].reshape(-1), target[x].reshape(-1)))
    return torch.tensor(jac_classwise)


def binary_acc(pred, target):
    pred_mask, target_mask = torch.zeros((256,256)), torch.zeros((256,256))
    for x in range(pred.shape[0]):
        pred_mask += ((pred[x] == 1).int() * x)
        target_mask += ((target[x] == 1).int() * x)
    
    return ((pred_mask == target_mask).sum()/256**2).item()


def metrics(pred, target):
    acc, f1, mcc = 1e-6, 1e-6, 1e-6
    for x in range(pred.shape[0]):
        cls_distr = ((pred[x] == target[x]).sum()/256**2).item()

        if cls_distr == 1:
            mcc += 1
            f1 += 1
            acc += 1
        if cls_distr == 0:
            mcc += 0
            f1 += 0
            acc += 0
        if cls_distr != 0 and cls_distr != 1:
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(pred[x].reshape(-1), target[x].reshape(-1)).ravel()   
            acc += (tp+tn)/(tp+tn+fp+fn)
            f1 += (2*(tp))/(2*(tp+fp+fn))
            mcc += ((tp*tn)-(fp*fn))/np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    
    return acc/pred.shape[0], f1/pred.shape[0], mcc/pred.shape[0], binary_acc(pred, target), jaccard(pred, target)



    
