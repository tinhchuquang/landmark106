import torch.nn as nn
import torchvision
import torch

class MobileNetV3Landmark(nn.Module):
    def __init__(self, num_points=106, pretrained=True):
        super().__init__()
        # Dùng backbone pretrain của torchvision
        base = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        self.backbone = base.features  # Feature extractor
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(576, num_points*2)  # 576 là out_channel cuối của mobilenet_v3_small
        # Nếu bạn dùng mobilenet_v3_large thì sửa fc in_features=960

    def forward(self, x):
        x = self.backbone(x)
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        return x


class Net(nn.Module):
    def __init__(self, model, num_points=106):
        super(Net, self).__init__()
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])

        self.fc = nn.Linear(25088, num_points*2)

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x