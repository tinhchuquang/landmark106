import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out

class HRModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, in_ch_list, out_ch_list):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList()
        for i in range(num_branches):
            layers = []
            for _ in range(num_blocks):
                layers.append(blocks(in_ch_list[i], out_ch_list[i]))
                in_ch_list[i] = out_ch_list[i]
            self.branches.append(nn.Sequential(*layers))
        self.fuse_layers = nn.ModuleList()
        for i in range(num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(num_branches):
                if i == j:
                    fuse_layer.append(nn.Identity())
                elif i < j:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(out_ch_list[j], out_ch_list[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(out_ch_list[i])
                    ))
                else:
                    ops = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            ops.append(nn.Sequential(
                                nn.Conv2d(out_ch_list[j], out_ch_list[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(out_ch_list[i])
                            ))
                        else:
                            ops.append(nn.Sequential(
                                nn.Conv2d(out_ch_list[j], out_ch_list[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(out_ch_list[j])
                            ))
                    fuse_layer.append(nn.Sequential(*ops))
            self.fuse_layers.append(fuse_layer)

    def forward(self, x_list):
        out_list = [branch(x) for branch, x in zip(self.branches, x_list)]
        fused = []
        for i in range(self.num_branches):
            y = out_list[i]
            for j in range(self.num_branches):
                if i == j:
                    continue
                elif i < j:
                    _, _, H, W = out_list[i].shape
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](out_list[j]), size=(H, W),
                        mode='bilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](out_list[j])
            fused.append(F.relu(y))
        return fused

class HRNetW18MultiScale(nn.Module):
    def __init__(self, num_landmarks=106, input_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.transition1 = nn.ModuleList([
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, 2, 1, bias=False),
                nn.BatchNorm2d(128), nn.ReLU(inplace=True)
            )
        ])
        self.stage2 = HRModule(
            num_branches=2, blocks=BasicBlock, num_blocks=2,
            in_ch_list=[64, 128], out_ch_list=[64, 128]
        )
        self.transition2 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, 2, 1, bias=False),
                nn.BatchNorm2d(256), nn.ReLU(inplace=True)
            )
        ])
        self.stage3 = HRModule(
            num_branches=3, blocks=BasicBlock, num_blocks=2,
            in_ch_list=[64, 128, 256], out_ch_list=[64, 128, 256]
        )
        self.transition3 = nn.ModuleList([
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
            nn.Sequential(
                nn.Conv2d(256, 512, 3, 2, 1, bias=False),
                nn.BatchNorm2d(512), nn.ReLU(inplace=True)
            )
        ])
        self.stage4 = HRModule(
            num_branches=4, blocks=BasicBlock, num_blocks=2,
            in_ch_list=[64, 128, 256, 512], out_ch_list=[64, 128, 256, 512]
        )

        # Head cho từng scale
        self.head_56 = nn.Conv2d(64, num_landmarks, 1)     # branch 0, 56x56
        self.head_28 = nn.Conv2d(128, num_landmarks, 1)    # branch 1, 28x28
        self.head_14 = nn.Conv2d(256, num_landmarks, 1)    # branch 2, 14x14

    def forward(self, x):
        x = self.stem(x)           # [B, 64, 56, 56]
        x1 = self.layer1(x)        # [B, 64, 56, 56]
        x_list = [self.transition1[0](x1), self.transition1[1](x1)]    # [B,64,56,56],[B,128,28,28]
        x_list = self.stage2(x_list)
        x_list = [
            self.transition2[0](x_list[0]),
            self.transition2[1](x_list[1]),
            self.transition2[2](x_list[1])
        ]
        x_list = self.stage3(x_list)
        x_list = [
            self.transition3[0](x_list[0]),
            self.transition3[1](x_list[1]),
            self.transition3[2](x_list[2]),
            self.transition3[3](x_list[2])
        ]
        x_list = self.stage4(x_list)

        out_56 = self.head_56(x_list[0])        # [B, 106, 56, 56]
        out_28 = self.head_28(x_list[1])        # [B, 106, 28, 28]
        out_14 = self.head_14(x_list[2])        # [B, 106, 14, 14]
        return out_56, out_28, out_14

if __name__ == "__main__":
    model = HRNetW18MultiScale(num_landmarks=106)
    x = torch.randn(2, 3, 224, 224)
    y1, y2, y3 = model(x)
    print('out_56:', y1.shape)   # [2, 106, 56, 56]
    print('out_28:', y2.shape)   # [2, 106, 28, 28]
    print('out_14:', y3.shape)   # [2, 106, 14, 14]