import torch
import torch.nn as nn
import torch.nn.functional as F

# --- C2f Block như YOLOv10 ---
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
        x = self.act(self.bn1(self.cv1(x)))
        for _ in range(self.n):
            x = self.act(self.bn2(self.cv2(x)))
        return x

# --- SPPF Block như YOLOv10 ---
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        self.cv1 = nn.Conv2d(c1, c2, 1, 1, bias=False)
        self.cv2 = nn.Conv2d(c2 * 4, c2, 1, 1, bias=False)
        self.bn  = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.k   = k

    def forward(self, x):
        x = self.act(self.cv1(x))
        y1 = F.max_pool2d(x, self.k, 1, self.k // 2)
        y2 = F.max_pool2d(y1, self.k, 1, self.k // 2)
        y3 = F.max_pool2d(y2, self.k, 1, self.k // 2)
        x  = torch.cat([x, y1, y2, y3], dim=1)
        return self.act(self.bn(self.cv2(x)))

# --- Neck: FPN-PAN (YOLOv10 đơn giản) ---
class YOLOv10FPN(nn.Module):
    def __init__(self, c3, c2, c1):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(c3, c2, 2, 2)
        self.up2 = nn.ConvTranspose2d(c2, c1, 2, 2)

    def forward(self, x3, x2, x1):
        u1 = self.up1(x3) + x2
        u2 = self.up2(u1) + x1
        return u2

# --- Model: YOLOv10 Landmark Coordinates (Flattened) ---
class YOLOv10_Landmark_Flatten(nn.Module):
    def __init__(self, num_landmarks=106):
        super().__init__()
        # Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.SiLU()
        )
        self.block1 = C2f(64, 128, n=2)
        self.block2 = C2f(128, 256, n=2)
        self.sppf   = SPPF(256, 256)
        # Neck
        self.fpn    = YOLOv10FPN(256, 128, 64)
        # Head: regression to 106*2 outputs, flattened -> 212
        self.head = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, num_landmarks * 2, 1, 1, 0)
        )

    def forward(self, x):
        x1 = self.stem(x)                             # [B,  64, 56, 56]
        x2 = self.block1(F.max_pool2d(x1, 2))         # [B, 128, 28, 28]
        x3 = self.block2(F.max_pool2d(x2, 2))         # [B, 256, 14, 14]
        x3 = self.sppf(x3)                            # [B, 256, 14, 14]
        feat = self.fpn(x3, x2, x1)                   # [B,  64, 56, 56]
        coords = self.head(feat)                      # [B, 212, 1, 1]
        coords = coords.view(coords.size(0), -1)      # [B, 212]
        return coords

if __name__ == "__main__":
    model = YOLOv10_Landmark_Flatten(num_landmarks=106)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)
    