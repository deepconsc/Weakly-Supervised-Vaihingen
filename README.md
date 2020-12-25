# Weakly-Supervised-Vaihingen

### This branch covers: 
- Standard multi-class classification to predict which classes are present in tile
- Multi-class labels are treated as soft labels. The value is generated in dataloader, where
we calculate the proportion between area covered by certain label masks over total area of image (tile).
- Class activation map probability density calculation over soft labels. We generate activation map of 
each class, and then calculate the density of pixels with probability over 90%. After all densities are 
calculated, we convert them to soft-label alike tensor, where we calculate the loss using BCE.
For future update, DICE loss might be a good way to go as well. 

For activation map calculation we have to break computational graph, but it's worth trying.

This commit is not fully tested yet. I have to run full trainer to check out some minimal bugs.


### TODO

- First run