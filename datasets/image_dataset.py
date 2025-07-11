import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataloader import LandmarkHeatmapDataset

def heatmaps_to_landmarks(heatmaps, img_size=224):
    """
    heatmaps: [num_points, H, W]
    return: [num_points, 2] tọa độ (x, y) trên ảnh resize (img_size x img_size)
    """
    num_points, h, w = heatmaps.shape
    coords = []
    for i in range(num_points):
        hm = heatmaps[i]
        # Lấy index giá trị max trên heatmap
        y, x = np.unravel_index(np.argmax(hm), hm.shape)
        # Scale về ảnh gốc
        x_img = x / w * img_size
        y_img = y / h * img_size
        coords.append([x_img, y_img])
    return np.array(coords)

# --- Example Usage ---
if __name__ == "__main__":
    img_dir = '/media/tinhcq/data1/Training_data/Lapa_Heatmap/train/images'
    label_dir = '/media/tinhcq/data1/Training_data/Lapa_Heatmap/train/landmarks'
    dataset = LandmarkHeatmapDataset([img_dir], [label_dir])
    img, heatmaps = dataset[0]  # img: torch.Size([3, 224, 224]), heatmaps: torch.Size([106, 56, 56])

    img_np = img.permute(1, 2, 0).numpy() * 255
    img_np = img_np.astype(np.uint8)
    landmarks = heatmaps_to_landmarks(heatmaps.numpy(), img_size=224)

    # Vẽ landmark lên ảnh
    for (x, y) in landmarks:
        cv2.circle(img_np, (int(x), int(y)), 2, (255, 0, 0), -1)

    plt.figure(figsize=(6,6))
    plt.imshow(img_np)
    plt.title('Landmarks from Heatmap')
    plt.axis('off')
    plt.show()
