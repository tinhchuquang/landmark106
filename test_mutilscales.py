import os
import torch
import cv2
import numpy as np
from utils.preproces import crop_center
from models.hrnet_w18_multiscale import HRNetW18MultiScale
from config.configs import num_points, image_size, std, mean
from scrfd import SCRFD
from utils.heatmap2landmark import heatmaps_to_landmarks

img_folder = '/data2/tinhcq/Training_data/Lapa_Heatmap/train/images'
model_weight = 'checkpoints/106_landmark/best_landmark106_hrnetw18.pth'

model = HRNetW18MultiScale(num_landmarks=106)
model.load_state_dict(torch.load(model_weight, map_location='cpu'))
model.eval()
detector = SCRFD(model_file='checkpoints/scrfd/scrfd_500m_bnkps.onnx')
detector.prepare(-1)

os.makedirs("test", exist_ok=True)
img_list = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in img_list:
    img_path = os.path.join(img_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Không đọc được ảnh: {img_path}")
        continue

    bboxes, kpss = detector.detect(img, 0.5, input_size=(640, 640))
    if len(bboxes) == 0:
        print(f"No face detected in {img_name}")
        continue
    box = bboxes[0][:4]
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center = np.array([center_x, center_y], dtype=np.float32)
    img_crop, (crop_x, crop_y) = crop_center(img, center, size=450)
    h0, w0 = img_crop.shape[:2]

    img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - mean) / std
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        pred_56, pred_28, pred_14 = model(img_tensor)  # 3 output

    # --- Landmark từng scale ---
    lm_56 = heatmaps_to_landmarks(pred_56)[0].cpu().numpy()
    lm_28 = heatmaps_to_landmarks(pred_28)[0].cpu().numpy()
    lm_14 = heatmaps_to_landmarks(pred_14)[0].cpu().numpy()
    # Scale về đúng kích thước ảnh crop
    lm_56[:, 0] = lm_56[:, 0] * w0 / 56
    lm_56[:, 1] = lm_56[:, 1] * h0 / 56
    lm_28[:, 0] = lm_28[:, 0] * w0 / 28
    lm_28[:, 1] = lm_28[:, 1] * h0 / 28
    lm_14[:, 0] = lm_14[:, 0] * w0 / 14
    lm_14[:, 1] = lm_14[:, 1] * h0 / 14

    # ========== (A) Weighted average ==========
    w1, w2, w3 = 0.5, 0.3, 0.2
    lm_ens = (w1 * lm_56 + w2 * lm_28 + w3 * lm_14) / (w1 + w2 + w3)

    # ========== (B) Confidence voting ==========
    # Lấy max value heatmap từng điểm trên từng scale
    conf_56 = pred_56[0].max(dim=(1, 2)).values.cpu().numpy()  # [106,]
    conf_28 = pred_28[0].max(dim=(1, 2)).values.cpu().numpy()
    conf_14 = pred_14[0].max(dim=(1, 2)).values.cpu().numpy()
    lm_stack = np.stack([lm_56, lm_28, lm_14], axis=0)  # [3, 106, 2]
    conf_stack = np.stack([conf_56, conf_28, conf_14], axis=0)  # [3, 106]
    best_idx = np.argmax(conf_stack, axis=0)
    lm_best = np.array([lm_stack[best_idx[i], i] for i in range(106)])  # [106, 2]

    # ========== Vẽ kết quả ==========
    img_vis = img_crop.copy()
    # Vẽ từng loại landmark (có thể comment nếu không cần)
    for (x, y) in lm_ens:   # Ensemble - Màu đỏ
        cv2.circle(img_vis, (int(x), int(y)), 2, (0, 0, 255), -1)
    for (x, y) in lm_best:  # Voting - Màu xanh lá
        cv2.circle(img_vis, (int(x), int(y)), 2, (0, 255, 0), -1)
    # Có thể vẽ thêm từng scale nếu thích...

    cv2.imwrite("test/" + img_name, img_vis)
    print(f"Đã lưu {img_name} (ensemble: đỏ, voting: xanh lá)")