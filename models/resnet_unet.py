import torch
import torch.nn as nn
from torchvision import models

class ResNetUNet(nn.Module):
    def __init__(self, n_classes=1):
        super(ResNetUNet, self).__init__()
        base_model = models.resnet34(weights=None)  
        base_layers = list(base_model.children())

        self.layer0 = nn.Sequential(*base_layers[:3])  # Conv1 + BN + ReLU
        self.layer1 = nn.Sequential(*base_layers[3:5]) # MaxPool + Layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.upconv4 = self._upsample(512, 256)
        self.upconv3 = self._upsample(256, 128)
        self.upconv2 = self._upsample(128, 64)
        self.upconv1 = self._upsample(64, 64)

        self.conv_last = nn.Conv2d(64, n_classes, kernel_size=1)

    def _upsample(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.upconv4(x4)
        x = self.upconv3(x)
        x = self.upconv2(x)
        x = self.upconv1(x)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        return self.conv_last(x)
