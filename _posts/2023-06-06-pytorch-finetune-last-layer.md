---
layout: post
title: "pytorch finetune last layer only"
date: 2023-06-06 18:11:00 +0800
tags: [pytorch,python]
---

## I have this resnet50 code and have a pretrained weight file to load. I would like to finetune the model only on the last fc layer, how to do that?
```python
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, stride0=2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=stride0, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride0, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = 512 * block.expansion
        self.fc = ChannelLinear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # transform form Pillow
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x
```

## ans by gpt4
In your ResNet50 model, you can freeze all the layers and only fine-tune the final fully connected layer (fc layer). To freeze all the layers, you just need to disable gradient computation for all the model parameters. This can be achieved by setting `requires_grad` attribute of each parameter to `False`.

Here's how you can do it:

```python
# Assume that `resnet` is your model
resnet = ResNet(YourBlock, [3, 4, 6, 3])  # replace YourBlock with the block you used

# Load the pre-trained weights
resnet.load_state_dict(torch.load('path_to_your_weights.pth'))

# Freeze all the layers
for param in resnet.parameters():
    param.requires_grad = False

# Unfreeze the last fully-connected layer
# `fc` is your last layer name
for param in resnet.fc.parameters():
    param.requires_grad = True

resnet = resnet.to(device)
```

Now, you can use this model for fine-tuning on your dataset. Remember to only use the parameters with `requires_grad = True` for the optimizer, which you can access with `resnet.parameters()`. This will not include the parameters of the frozen layers in the computation of gradients, effectively "freezing" them:

```python
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, resnet.parameters()), lr=0.001, momentum=0.9)
```

Note that `filter(lambda p: p.requires_grad, resnet.parameters())` will only consider the parameters where `requires_grad=True`, which are the parameters of the last fc layer in this case.