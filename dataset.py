import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from torchvision import transforms

class Landmark106Dataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=224, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.transform = transform
        self.samples = []
        for fname in os.listdir(self.img_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                label_name = fname.rsplit('.', 1)[0] + ".txt"
                if os.path.exists(os.path.join(self.label_dir, label_name)):
                    self.samples.append((fname, label_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, label_name = self.samples[idx]
        img_path = os.path.join(self.img_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        image = cv2.resize(image, (self.img_size, self.img_size))
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image.transpose(2,0,1), dtype=torch.float32) / 255.0
        # Load 106 pts
        points = []
        with open(label_path, 'r') as f:
            for line in f:
                x, y = map(float, line.strip().split())
                # Nếu label là pixel: chuẩn hóa về [0,1]
                x_norm = x / w
                y_norm = y / h
                points.extend([x_norm, y_norm])
        return image, torch.tensor(points, dtype=torch.float32)