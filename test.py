import os
import torch
import cv2
import numpy as np
from utils.preproces import  crop_center
# from models.hrnet import get_face_alignment_net
# from config.hrnet_w18_config import config as hrnet_w18_config
from models.hrnet_w18 import HRNetW18
from config.configs import num_points, image_size, std, mean
from scrfd import SCRFD
from utils.heatmap2landmark import heatmaps_to_landmarks

# ==== Đường dẫn ====
img_folder = '/data2/tinhcq/Training_data/Lapa_Heatmap/train/images'  # Thư mục chứa ảnh
model_weight = 'checkpoints/106_landmark/best_landmark106_hrnetw18.pth'

# ==== Load model Landmark ====
model = HRNetW18(num_landmarks=106)
model.load_state_dict(torch.load(model_weight, map_location='cpu'))
model.eval()
detector = SCRFD(model_file='checkpoints/scrfd/scrfd_500m_bnkps.onnx')
detector.prepare(-1)


# ==== Lấy danh sách ảnh ====
img_list = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_name in img_list:
    img_path = os.path.join(img_folder, img_name)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Không đọc được ảnh: {img_path}")
        continue

    bboxes, kpss = detector.detect(img, 0.5, input_size=(640, 640))
    box = bboxes[0][:4]  # lấy bbox đầu tiên (hoặc bạn chọn bbox có score lớn nhất)
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    center = np.array([center_x, center_y], dtype=np.float32)
    img_crop, (crop_x, crop_y) = crop_center(img, center, size=450)
    # img_crop = img.copy()
    h0, w0 = img_crop.shape[:2]

    # Tiền xử lý
    img_rgb = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (image_size, image_size))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = (img_norm - mean) / std
    img_tensor = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).float()

    # Inference
    with torch.no_grad():
        pred = model(img_tensor)
    # Đưa về pixel trên ảnh gốc
    landmark = heatmaps_to_landmarks(pred)[0].cpu()
    landmark[:, 0] = landmark[:, 0] * w0 / 56
    landmark[:, 1] = landmark[:, 1] * h0 / 56

    # Vẽ landmark lên ảnh gốc (BGR)
    img_vis = img_crop.copy()
    for (x, y) in landmark:
        cv2.circle(img_vis, (int(x), int(y)), 2, (0, 0, 255), -1)

    # Hiển thị bằng OpenCV
    cv2.imwrite("test/" + img_name, img_vis)
    # key = cv2.waitKey(0)
    # if key == 27:  # ESC để thoát
    #     break

# cv2.destroyAllWindows()
