---
layout: post
title: "PyTorch image data visualization"
date: 2023-06-05 20:10:00 +0800
tags: [pytorch, python]
---

## how to explore and visualise 3 inputs from the training data

You can visualize your input images using libraries like Matplotlib or OpenCV. 

Here's an example of how you can visualize 3 input images from your training data using PyTorch and Matplotlib:

```python
import matplotlib.pyplot as plt
import torchvision

# function to unnormalize and convert tensor to numpy for visualization
def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()

# Get a batch of training data
inputs, classes = next(iter(train_loader))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs[:3])

imshow(out)
```

In this script:

1. `imshow` is a helper function that unnormalizes an image and converts it from a Tensor to a numpy array so it can be displayed by Matplotlib.

2. `inputs, classes = next(iter(train_loader))` gets a batch of training data. `inputs` is a batch of input images, and `classes` are the labels for those images.

3. `out = torchvision.utils.make_grid(inputs[:3])` takes the first three images from the batch and combines them into a single image that can be displayed.

4. `imshow(out)` displays the grid of images.

This script assumes that your images have been normalized using the mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`, and that they are loaded using a DataLoader. Adjust it as necessary to match your setup.

Note that this code will work if your images are color (RGB) images. If your images are grayscale or if they have a different color format, you might need to modify the code accordingly.