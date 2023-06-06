---
layout: post
title: "PyTorch decaying lr"
date: 2023-06-06 18:12:00 +0800
tags: [pytorch,python]
---

## pytorch optimizer with decaying learning rate every 10000 steps

Sure, one of the common ways to apply a decaying learning rate in PyTorch is to use the built-in learning rate scheduler. This can be done using the `torch.optim.lr_scheduler` module. 

There are multiple types of schedulers provided by PyTorch such as `StepLR`, `ExponentialLR`, `MultiStepLR` and `CosineAnnealingLR` among others. However, none of these directly support decay every specific number of steps. 

For your specific use case, `StepLR` can be used with some modifications in your training loop. The `StepLR` class decays the learning rate every specified number of epochs, so if you want to decay every 10000 steps, you would need to calculate how many epochs that would be based on the size of your dataset and your batch size.

Alternatively, `LambdaLR` can be used to create a custom learning rate scheduler. Here is an example:

```python
import torch
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

model = ...
optimizer = SGD(model.parameters(), lr=1e-3)

# Define a lambda function that takes the current step `i` and returns the multiplier for the learning rate
# This will decrease the learning rate by half every 10000 steps
lr_lambda = lambda i: 0.5 ** (i // 10000)

# Pass the lambda function to the LambdaLR scheduler
scheduler = LambdaLR(optimizer, lr_lambda)

for epoch in range(num_epochs):
    for i, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Step the scheduler
        scheduler.step()
```

Please note that this code example assumes that you are using a batch size of 1. If you are using a larger batch size, you should call `scheduler.step()` after each batch instead of after each sample. Please also ensure to adjust the lambda function to consider your batch size if it is greater than 1.
