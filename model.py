import torch.nn as nn
from torchvision import models

# class LandmarkModel(nn.Module):
#     def __init__(self, num_points=106):
#         super().__init__()
#         self.backbone = models.mobilenet_v2(pretrained=True).features
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(1280, num_points*2)
#
#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.gap(x).flatten(1)
#         x = self.fc(x)
#         return x
#
#

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super().__init__()
        reduced = max(1, int(in_channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, reduced, 1)
        self.fc2 = nn.Conv2d(reduced, in_channels, 1)

    def forward(self, x):
        se = self.pool(x)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se))
        return x * se

class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3) / 6


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, exp_channels, se=False, nl='RE'):
        super().__init__()
        self.use_res_connect = (stride == 1 and in_channels == out_channels)
        act = nn.ReLU if nl == 'RE' else HSwish

        layers = []
        if exp_channels != in_channels:
            layers.append(nn.Conv2d(in_channels, exp_channels, 1, bias=False))
            layers.append(nn.BatchNorm2d(exp_channels))
            layers.append(act())
        # DWConv
        layers.append(nn.Conv2d(exp_channels, exp_channels, kernel_size, stride, kernel_size//2, groups=exp_channels, bias=False))
        layers.append(nn.BatchNorm2d(exp_channels))
        layers.append(act())
        # Squeeze-Excite
        if se:
            layers.append(SEBlock(exp_channels))
        # Pointwise linear
        layers.append(nn.Conv2d(exp_channels, out_channels, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MobileNetV3Small(nn.Module):
    def __init__(self, num_points=106):
        super().__init__()
        self.num_points = num_points
        self.act = HSwish

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),   # 112x112
            nn.BatchNorm2d(16),
            HSwish(),
        )
        # Stage
        self.blocks = nn.Sequential(
            MBConv(16, 16, 3, 2, 16,  True, 'RE'),      # 56x56
            MBConv(16, 24, 3, 2, 72,  False, 'RE'),     # 28x28
            MBConv(24, 24, 3, 1, 88,  False, 'RE'),
            MBConv(24, 40, 5, 2, 96,  True, 'HS'),      # 14x14
            MBConv(40, 40, 5, 1, 240, True, 'HS'),
            MBConv(40, 40, 5, 1, 240, True, 'HS'),
            MBConv(40, 48, 5, 1, 120, True, 'HS'),      # 14x14
            MBConv(48, 48, 5, 1, 144, True, 'HS'),
            MBConv(48, 96, 5, 2, 288, True, 'HS'),      # 7x7
            MBConv(96, 96, 5, 1, 576, True, 'HS'),
            MBConv(96, 96, 5, 1, 576, True, 'HS'),
        )
        # Last part
        self.conv_last = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            HSwish()
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(576, 1024)
        self.act_fc = HSwish()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1024, num_points * 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.conv_last(x)
        x = self.pool(x).flatten(1)
        x = self.fc1(x)
        x = self.act_fc(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    model = MobileNetV3Small(num_points=106)
    x = torch.randn(2, 3, 224, 224)  # batch size 2
    y = model(x)
    print(y.shape)  #