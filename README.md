# Weakly-Supervised-Vaihingen

### This branch covers: 
- Standard multi-class classification to predict which classes are present in tile
- Multi-class labels are treated as soft labels. The value is generated in dataloader, where
we calculate the proportion between area covered by certain label masks over total area of image (tile).
- Class activation map probability density calculation over soft labels. We generate activation map of 
each class, and then calculate the density of pixels with probability over 90%. After all densities are 
calculated, we convert them to soft-label alike tensor, where we calculate the loss using L1 loss.
For future update, DICE loss might be a good way to go as well. 
- We use BCE for soft labels, L1 for activation map probability density.

For activation map calculation we have to break computational graph, but it's worth trying.

### First iteration

- Rewrote the patch generator & recoverer algorithms.
- The trainer runs smoothly.
- Adam works better than SGD.
- Losses are decreasing.
- Trainer is incredebly fast. Well, most of it comes from
Resnet50 and 200x200 patches, but we're doing a lot of pre & post processing.
- Needs heavier augmentation.


### TODO

- Code the visualization tool. Visual input would be much informative in this case.
