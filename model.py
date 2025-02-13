import torch
from torch import nn
from torchvision import models


class ResNetEncoder(nn.Module):
    def __init__(self, resnet_model):
        super(ResNetEncoder, self).__init__()
        self.resnet = resnet_model

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)

        return x1, x2, x3, x4


class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UNetDecoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.size(2) != skip.size(2) or x.size(3) != skip.size(3):
            x = nn.functional.interpolate(x, size=(skip.size(2), skip.size(3)), mode='bilinear', align_corners=True)
        x = torch.cat((x, skip), dim=1)
        x = self.conv(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, num_classes):
        super(ResUNet, self).__init__()
        resnet = models.resnet34(pretrained=True)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder = ResNetEncoder(resnet)
        self.decoder1 = UNetDecoder(512, 256, 256)
        self.decoder2 = UNetDecoder(256, 128, 128)
        self.decoder3 = UNetDecoder(128, 64, 64)
        self.decoder4 = UNetDecoder(64, 64, 64)
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        d1 = self.decoder1(x4, x3)
        d2 = self.decoder2(d1, x2)
        d3 = self.decoder3(d2, x1)
        d4 = self.decoder4(d3, x1)
        out = self.final_conv(d4)

        # 插值操作, 将输出分辨率调整为输入分辨率
        out = nn.functional.interpolate(out, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=True)
        return out
