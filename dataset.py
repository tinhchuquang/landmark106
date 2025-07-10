import os
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from config.configs import image_size, img_dir, label_dir

class Landmark106Dataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=224, transform=None, augment=True):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.samples = []
        for img_folder, label_folder in zip(self.img_dir, self.label_dir):
            for fname in os.listdir(label_folder):
                label_path = os.path.join(label_folder, fname)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                num_points = len(lines) - 1  # bỏ dòng đầu
                if num_points != 106:
                    continue
                image_name = fname.replace('.txt', '.jpg')
                img_path = os.path.join(img_folder, image_name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, label_path))
                else:
                    print(f"Không tìm thấy ảnh: {img_path} cho label: {label_path}")

        # Albumentations pipeline
        self.transform = transform
        if self.transform is None and augment:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.8),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
        elif self.transform is None:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2(),
            ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        # Load 106 pts
        points = []
        with open(label_path, 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                x, y = map(float, line.strip().split())
                points.append((x, y))  # dùng dạng tuple cho albumentations

        # Apply albumentations
        transformed = self.transform(image=image, keypoints=points)
        image = transformed['image']
        keypoints = transformed['keypoints']  # list of (x, y) sau augment
        # Chuẩn hóa về [0,1] so với img_size
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        keypoints = keypoints / self.img_size  # (N,2)
        keypoints = keypoints.flatten()  # [212]
        return image, keypoints

if __name__ == '__main__':
    # img_dir = '/media/tinhcq/data1/Training_data'
    # label_dir = '/media/tinhcq/data1/Training_data/Corrected_landmark/Corrected_landmark'
    # img_size = 224

    dataset = Landmark106Dataset(img_dir, label_dir, img_size=image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, (img, pts) in enumerate(dataloader):
        print(f"Image shape: {img.shape}")  # [1, 3, 224, 224]
        print(f"Landmark shape: {pts.shape}")  # [1, 212]
        print("Landmark points (normalized):", pts[0][:10].numpy())  # Print 5 points đầu
        # Hiển thị ảnh + landmark (denormalize về 224)
        img_vis = img[0].permute(1, 2, 0).cpu().numpy()
        img_vis = (img_vis * 0.229 + 0.485) * 255  # Approx de-normalize
        img_vis = np.clip(img_vis, 0, 255).astype('uint8')
        xs = pts[0][0::2].numpy() * image_size
        ys = pts[0][1::2].numpy() * image_size
        plt.figure(figsize=(5,5))
        plt.imshow(img_vis)
        plt.scatter(xs, ys, s=8, c='red')
        plt.title('Sample with landmarks')
        plt.show()
        # break