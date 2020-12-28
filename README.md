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
- U^2-Net (U Squared Net) is used for segmentation. We only chage input channels 3 -> 4. 
- Pix2Pix training approach with its original discriminator is used. We additionally use BCE loss over U^2-Net intermediate
layers' outputs as auxiliary loss for Generator.


### First iteration

- Pix2Pix trainer is finished.
- U^2-Net is implemented.
- Auxiliary loss is implemented.
- Network runs smoothly, but it's slow. We're moving 5 channel target masks back and forth, that's why.
- The results on 5th epoch are very much impressive.

### Predictions
![Alt text](images/epoch_5_1.png?raw=true "5th epoch, Batch 1. Upper - Predicted, Lower - Target")
![Alt text](images/epoch_5_2.png?raw=true "5th epoch, Batch 2. Upper - Predicted, Lower - Target")
![Alt text](images/epoch_5_3.png?raw=true "5th epoch, Batch 3. Upper - Predicted, Lower - Target")



### TODO

- Train network for around 100 epochs.