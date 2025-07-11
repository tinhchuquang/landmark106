import torch
import torch.nn as nn
import torch.nn.functional as F

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.cv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.n = n

    def forward(self, x):
        x = self.cv1(x)
        x = self.bn1(x)
        x = self.act(x)
        for _ in range(self.n):
            x = self.cv2(x)
            x = self.bn2(x)
            x = self.act(x)
        return x

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c2 * 4, c2, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.k = k

    def forward(self, x):
        x = self.cv1(x)
        y1 = F.max_pool2d(x, self.k, 1, self.k // 2)
        y2 = F.max_pool2d(y1, self.k, 1, self.k // 2)
        y3 = F.max_pool2d(y2, self.k, 1, self.k // 2)
        x = torch.cat([x, y1, y2, y3], 1)
        x = self.cv2(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
class YOLOv10FPN(nn.Module):
    def __init__(self, c3, c2, c1):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.up2 = nn.ConvTranspose2d(c2, c1, 2, 2)

    def forward(self, x3, x2, x1):
        u1 = self.up1(x3) + x2  # [B, c2, 28, 28]
        u2 = self.up2(u1) + x1  # [B, c1, 56, 56]
        return u2, u1, x3        # return luôn cả 3 scale

class YOLOv10_Landmark_MultiScale(nn.Module):
    def __init__(self, num_landmarks=106):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU()
        )
        self.block1 = C2f(64, 128, n=2)
        self.block2 = C2f(128, 256, n=2)
        self.sppf = SPPF(256, 256)
        self.fpn = YOLOv10FPN(256, 128, 64)

        # 3 head cho 3 scale, mỗi head là Conv2d (in_ch, num_landmarks, 1)
        self.head_small = nn.Conv2d(64, num_landmarks, 1)    # P3 [B, 106, 56, 56]
        self.head_medium = nn.Conv2d(128, num_landmarks, 1)  # P4 [B, 106, 28, 28]
        self.head_large = nn.Conv2d(256, num_landmarks, 1)   # P5 [B, 106, 14, 14]

    def forward(self, x):
        x = self.stem(x)
        x1 = x                                  # [B, 64, 56, 56]
        x2 = self.block1(F.max_pool2d(x1, 2))   # [B, 128, 28, 28]
        x3 = self.block2(F.max_pool2d(x2, 2))   # [B, 256, 14, 14]
        x3 = self.sppf(x3)                      # [B, 256, 14, 14]

        feat_small, feat_medium, feat_large = self.fpn(x3, x2, x1)  # [B, 64, 56,56], [B,128,28,28], [B,256,14,14]

        out_small = self.head_small(feat_small)      # [B, 106, 56, 56]
        out_medium = self.head_medium(feat_medium)   # [B, 106, 28, 28]
        out_large = self.head_large(feat_large)      # [B, 106, 14, 14]
        return [out_small, out_medium, out_large]

if __name__ == "__main__":
    model = YOLOv10_Landmark_MultiScale(num_landmarks=106)
    x = torch.randn(4, 3, 224, 224)
    outs = model(x)
    for o in outs:
        print(o.shape)
    # out_small: [2, 106, 56, 56]
    # out_medium: [2, 106, 28, 28]
    # out_large: [2, 106, 14, 14]