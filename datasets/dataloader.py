import torch
import os
import numpy as np
import cv2
from torch.utils.data import Dataset
from utils.preproces import crop_center

def load_landmark_txt(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    num_points = int(lines[0].strip())
    center = np.array(list(map(float, lines[1].strip().split())))
    points = []
    for line in lines[2:2+num_points]:
        x, y = map(float, line.strip().split())
        points.append([x, y])
    points = np.array(points, dtype=np.float32)
    return center, points

def draw_gaussian(heatmap, center, sigma):
    tmp_size = sigma * 3
    mu_x = int(center[0])
    mu_y = int(center[1])
    w, h = heatmap.shape[1], heatmap.shape[0]

    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

    if ul[0] >= w or ul[1] >= h or br[0] < 0 or br[1] < 0:
        return heatmap

    size = 2 * tmp_size + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2

    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    g_x = max(0, -ul[0]), min(br[0], w) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], h) - ul[1]

    img_x = max(0, ul[0]), min(br[0], w)
    img_y = max(0, ul[1]), min(br[1], h)

    heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]],
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    )
    return heatmap

def generate_heatmaps(points, heatmap_size=(56, 56), img_size=224, sigma=2):
    num_points = points.shape[0]
    heatmaps = np.zeros((num_points, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
    for i in range(num_points):
        pt = points[i]
        hm_x = pt[0] / img_size * heatmap_size[0]
        hm_y = pt[1] / img_size * heatmap_size[1]
        heatmaps[i] = draw_gaussian(heatmaps[i], (hm_x, hm_y), sigma)
    return heatmaps



class LandmarkHeatmapDataset(Dataset):
    def __init__(
        self,
        img_dir,
        label_dir,
        img_size=224,
        heatmap_size=(56, 56),
        sigma=2,
        crop_size=450,
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.crop_size = crop_size
        self.samples = []
        for img_folder, label_folder in zip(self.img_dir, self.label_dir):
            for fname in os.listdir(label_folder):
                img_name = fname.replace('.txt', '.jpg')
                img_path = os.path.join(img_folder, img_name)
                label_path = os.path.join(label_folder, fname)
                if os.path.exists(img_path):
                    self.samples.append((img_path, label_path))
        # Augmentation pipeline (bạn có thể tuỳ chỉnh thêm transform nếu muốn)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        center, points = load_landmark_txt(label_path)
        img_crop, (crop_x, crop_y) = crop_center(img, center, size=self.crop_size)
        points_crop = points - np.array([crop_x, crop_y])


        # Resize về img_size x img_size (ảnh và landmark)
        img_resized = cv2.resize(img_crop, (self.img_size, self.img_size))
        scale_x = self.img_size / img_crop.shape[1]
        scale_y = self.img_size / img_crop.shape[0]
        points_resized = points_crop.copy()
        points_resized[:, 0] *= scale_x
        points_resized[:, 1] *= scale_y

        img_tensor = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1)

        heatmaps = generate_heatmaps(points_resized, heatmap_size=self.heatmap_size, img_size=self.img_size, sigma=self.sigma)
        heatmaps = torch.from_numpy(heatmaps)

        return img_tensor, heatmaps

# --- Example Usage ---
if __name__ == "__main__":
    img_dir = '/media/tinhcq/data1/Training_data/Lapa_Heatmap/train/images'
    label_dir = '/media/tinhcq/data1/Training_data/Lapa_Heatmap/train/landmarks'
    dataset = LandmarkHeatmapDataset(img_dir, label_dir)
    img, heatmaps = dataset[0]
    print(img.shape, heatmaps.shape)  # [3, 224, 224], [106, 56, 56]