'''
-----------PBI 2020----------------
@author: cyborg
@module: VGG
@library: PyTorch
'''

import torch
import torch.nn as nn

VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 
         512, 512, 512, 'M', 512, 512, 512, 'M']

class VGGarch(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(VGGarch, self).__init__()
        self.in_channels = in_channels
        self.conv_layers = self._layers(VGG16)
        self.fc = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
            )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    
    def _layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        
        for x in VGG16:
            if type(x) == int:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                     kernel_size=3, padding=1, stride=1),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU()]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            
        return nn.Sequential(*layers)
    
model = VGGarch(in_channels=3, num_classes=1000)
x = torch.randn(1,3,224,224)
print(model(x).shape)