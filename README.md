# Weakly-Supervised-Vaihingen

### Sum up:
Pix2Pix network with U^2-Net generator with encoder weak pretraining and Monocular Depth Estimation network output as fourth spatial dimension.

### This branch covers: 
- Random patch cropping function. Basically we take center coordinate with enough distance from corners and then expand
box proportionally in x & y directions. This is a better working approach for low-resource semantic segmentation, because 
we'll be having more disturbed data over original distribution - i.e. natural augmentation, and we can generate much more
patches (cherry on top).
- Monocular Depth Estimation network is used to generate depth map. We stack depth map as 4th channel to original RGB image, 
which goes through the U^2-Net network. I find this very much intuitive over raw RGB approach, as long as it's additional dense 
spatial information. Actually depth map showed promising results even while operating on raw single channel depth map. It 
estimates corners very well, it only struggles near 'Car' class, where we can hardly see class mask objectively. 
- U^2-Net (U Squared Net) is used as Generator. We chage input channels 3 -> 4, and add 2x Conv, Batchnorm, Maxpool
layers along with Linear to support Encoder + Classification during pretraining.
- Pix2Pix training approach with its original discriminator is used. We additionally use BCE loss over U^2-Net intermediate
layers' outputs as auxiliary loss for Generator.
- U^2-Net Encoder pretraining as multi-label classifier on tile-level crop labels in weakly supervised manner.


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

- Converted U^2-Net as Encoder + classifier + Decoder. Pretrained it on 23 HR images' tile-level class labels.
- Took classifier away & trained on 3 HR images' full-pixel masks.

# Reported Metrics on Last iteration:
IoU is pure jaccard similarity score.
Binary ACC is by conversion of 5-dim output mask to 1-dim mask. This way we 
don't have to tackle with false accuracy scores, when background is huge in
many images. 

In this case, Binary ACC is the most precise total class mask accuracy. 

The validation is done on 7 unseen HR images in the dataset.

```
Mean IoU: 94.95
F1: 0.67
Binary ACC: 0.82
Classwise IoU: 
0 - 93.27
1 - 97.84
2 - 91.46
3 - 92.63
4 - 99.57
```

### Predictions
Upper - Predicted, Lower - Target.
Images below are all 5 classes 0 -> 4, from left to right respectively.

**20th Epoch**
- Batch 1. 
![Alt text](images/epoch_20_1.png?raw=true " ")
- Batch 2.
![Alt text](images/epoch_20_2.png?raw=true " ")
- Batch 3. 
![Alt text](images/epoch_20_3.png?raw=true " ")


**Last Epoch**
- Batch 1. 
![Alt text](images/epoch_200_1.png?raw=true " ")
- Batch 2.
![Alt text](images/epoch_200_2.png?raw=true " ")
- Batch 3. 
![Alt text](images/epoch_200_3.png?raw=true " ")
- Batch 4. 
![Alt text](images/epoch_200_4.png?raw=true " ")

