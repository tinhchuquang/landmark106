import os
import torch
import cv2
import numpy as np
from model import MobileNetV3Landmark
from config.configs import num_points, image_size, std, mean

# ==== Đường dẫn ====
img_folder = '/media/tinhcq/data1/Training_data/LaPa/val/images'  # Thư mục chứa ảnh
model_weight = 'checkpoints/106_landmark/best_landmark106_yolo.pth'

# ==== Load model Landmark ====
model = YOLOv10Landmark(num_points=num_points)
model.load_state_dict(torch.load(model_weight, map_location='cpu'))
# model.eval()

# ==== Lấy danh sách ảnh ====
img_list = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in img_list:
    img_path = os.path.join(img_folder, img_name)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không đọc được ảnh: {img_path}")
        continue
    h0, w0 = img.shape[:2]

    # Tiền xử lý
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - mean) / std
    img_tensor = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).float()

    # Inference
    with torch.no_grad():
        pred = model(img_tensor)  # [1, 212]
        pred = pred.view(num_points, 2).cpu().numpy()
    # Đưa về pixel trên ảnh gốc
    pred[:, 0] = pred[:, 0] * w0
    pred[:, 1] = pred[:, 1] * h0

    # Vẽ landmark lên ảnh gốc (BGR)
    img_vis = img.copy()
    for (x, y) in pred:
        cv2.circle(img_vis, (int(x), int(y)), 2, (0, 0, 255), -1)

    # Hiển thị bằng OpenCV
    cv2.imshow("Landmark 106", img_vis)
    key = cv2.waitKey(0)
    if key == 27:  # ESC để thoát
        break

cv2.destroyAllWindows()
