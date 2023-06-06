---
layout: post
title: "pytorch foldername as data label"
date: 2023-06-05 22:12:00 +0800
tags: [pytorch, python]
---

## python CNN model using pytorch, load input data label according to folder name

Sure, here is a general template for a Convolutional Neural Network (CNN) in PyTorch. This example assumes you have a dataset of images organized in folders where each folder name corresponds to the label of the images inside it.

To load the data according to the folder names, you can use the `ImageFolder` class from `torchvision.datasets`. 

Here is the full code:

```python
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

# Define your transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load your dataset
dataset = torchvision.datasets.ImageFolder(root='path_to_your_dataset', transform=transform)

# Split your dataset into training and validation sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create your DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

# Define your CNN architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(dataset.classes))  # Number of classes is equal to the number of folders

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize your model
net = Net()

# Choose a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train your model
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
```

Please replace `'path_to_your_dataset'` with the actual path of your dataset. Also, the CNN

## shall we add `img = Image.open(filename).convert('RGB')` to our transform.compose?

The `convert('RGB')` method is used to convert an image to the RGB color space. This is typically done because some images might be in different color spaces (like RGBA, which includes an alpha channel for transparency, or grayscale) and the model expects the input images to be in the RGB color space.

You should include this conversion if your dataset contains images that are not in RGB format. However, when using the PyTorch torchvision transforms, the conversion to tensor with `transforms.ToTensor()` will also convert the image to the RGB format, so you typically don't need to explicitly include `convert('RGB')`.

But to ensure that all images are in RGB format before applying transformations, you can use a custom transform with the `PIL.Image` library. Here is how you can do it:

```python
from PIL import Image

class ConvertRGB:
    def __call__(self, img):
        return img.convert('RGB')

# Then add it to your transformations
transform = transforms.Compose([
    ConvertRGB(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

This way, the `ConvertRGB` transform will ensure that all images are in RGB format before the other transforms are applied. Note that this is generally not necessary when using `ImageFolder` from torchvision, as it should handle images in various formats correctly. You only need to add it if you have images in different formats and encounter issues without it.