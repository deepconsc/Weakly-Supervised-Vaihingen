# Weakly-Supervised-Vaihingen

### Sum up:
Pix2Pix network with U^2-Net generator and Monocular Depth Estimation network output as fourth spatial dimension.

### This branch covers: 
- Random patch cropping function. Basically we take center coordinate with enough distance from corners and then expand
box proportionally in x & y directions. This is a better working approach for low-resource semantic segmentation, because 
we'll be having more disturbed data over original distribution - i.e. natural augmentation, and we can generate much more
patches (cherry on top).
- Monocular Depth Estimation network is used to generate depth map. We stack depth map as 4th channel to original RGB image, 
which goes through the U^2-Net network. I find this very much intuitive over raw RGB approach, as long as it's additional dense 
spatial information. Actually depth map showed promising results even while operating on raw single channel depth map. It 
estimates corners very well, it only struggles near 'Car' class, where we can hardly see class mask objectively. 
- U^2-Net (U Squared Net) is used as Generator. We chage input channels 3 -> 4, and add couple of Conv, Batchnorm, Maxpool
layers along with Linear to support Encoder + Classification during pretraining.
- Pix2Pix training approach with its original discriminator is used. We additionally use BCE loss over U^2-Net intermediate
layers' outputs as auxiliary loss for Generator.


### First iteration

- Pix2Pix trainer is finished.
- U^2-Net is implemented.
- Auxiliary loss is implemented.
- Network runs smoothly, but it's slow. We're moving 5 channel target masks back and forth, that's why.
- The results on 5th epoch are very much impressive.

### Second Iteration

- The results are stable and good. U^2-Net reached near-perfect results for around 50th epoch 
with adversarial learning.
- Added samples of 50th epoch.

### Third Iteration

- Training is almost finished, 180 epochs.
- Added validation script of unseen 7 HR images.


### Final Iteration

- Converted U^2-Net as Encoder + classifier. Pretrained it on 23 HR images' tile-level class labels.
- Took classifier away & trained on 3 HR images' full-pixel masks.
- Metrics report is coming soon.

### Reported Metrics on 3rd iteration:
IoU is pure jaccard similarity score.
Binary ACC is by conversion of 5-dim output mask to 1-dim mask. This way we 
don't have to tackle with false accuracy scores, when background is huge in
many images. 

In this case, Binary ACC is the most precise total class mask accuracy. 

The validation is done on 7 unseen HR images in the dataset.

```
Mean IoU: 91.50
Accuracy: 91.50
F1: 0.55
Binary ACC: 0.78
Classwise IoU: 
0 - 85.20
1 - 95.41
2 - 85.73
3 - 91.95
4 - 99.20
```

### Predictions
Upper - Predicted, Lower - Target.
Images below are all 5 classes 0 -> 4, from left to right respectively.

**5th Epoch**
- Batch 1. 
![Alt text](images/epoch_5_1.png?raw=true " ")
- Batch 2.
![Alt text](images/epoch_5_2.png?raw=true " ")
- Batch 3. 
![Alt text](images/epoch_5_3.png?raw=true " ")


**50th Epoch**
- Batch 1. 
![Alt text](images/epoch_50_1.png?raw=true " ")
- Batch 2.
![Alt text](images/epoch_50_2.png?raw=true " ")
- Batch 3. 
![Alt text](images/epoch_50_3.png?raw=true " ")
- Batch 4. 
![Alt text](images/epoch_50_4.png?raw=true " ")

