# Monocular Depth Prediction with a Fully Convolutional Neural Network and Skip Connections

![](Prediction%20Samples.png?raw=true)

### Abstract
This work tackles the problem of estimating depth from a single RGB image of a scene. We propose a fully convolutional neural network that uses skip connections between layers to capture the relationship between an image and its depth. Our model learns both local and global features through its U-shaped architecture, allowing it to successfully find a consistent depth for the entire image. We also send activations from previous layers to the following layers in “skip connections,” which helps the architecture preserve the overall structure of the image. After our end-to-end training process, we evaluate on a variety of losses and accuracies and find that our model performs similar to or better than many past architectures trained on the same dataset while requiring fewer computational resources and less training time.

### Architecture
![](Diagram.png?raw=true)

### Results
| RMS | Rel | log10 | Acc (d=1.25) | Acc (d=1.25^2) | Acc (d=1.25^3) |
| --- | --- | ----- | ------------ | -------------- | -------------- |
| 0.8173 | 0.2458 | 0.09698 | 0.6018 | 0.8833 | 0.9672 |
