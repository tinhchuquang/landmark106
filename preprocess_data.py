import cv2
import numpy as np
import os
from scrfd import SCRFD

if __name__ == '__main__':
    detector = SCRFD(model_file='checkpoints/scrfd/scrfd_500m_bnkps.onnx')
    detector.prepare(-1)
    label_dir = '/media/tinhcq/data1/Training_data/Corrected_landmark/Corrected_landmark'
    img_dir = '/media/tinhcq/data1/Training_data'

    samples = []
    for fname in os.listdir(label_dir):
        label_path = os.path.join(label_dir, fname)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        num_points = len(lines) - 1
        if num_points != 106:
            continue
        image_name = fname.split('_')[0] + '/picture/' + fname.replace('.txt', '')
        img_path = os.path.join(img_dir, image_name)
        if os.path.exists(img_path):
            samples.append((img_path, label_path))

    path_save = '/media/tinhcq/data1/Training_data/Data'
    os.makedirs(os.path.join(path_save, 'image'), exist_ok=True)
    os.makedirs(os.path.join(path_save, 'label'), exist_ok=True)

    for s in samples:
        points = []
        image = cv2.imread(s[0])
        with open(s[1], 'r') as f:
            lines = f.readlines()[1:]
            for line in lines:
                x, y = map(float, line.strip().split())
                points.append([x, y])
        points = np.array(points, dtype=np.float32)
        #
        # bboxes, kpss = detector.detect(image, 0.5, input_size=(640, 640))
        # if len(bboxes) == 0:
        #     print(f"Không detect được face: {s[0]}")
        #     continue
        #
        # # Chọn bbox đầu tiên (lớn nhất)
        # x1, y1, x2, y2, score = bboxes[0]
        # w = x2 - x1
        # h = y2 - y1
        #
        # # Mở rộng bbox
        # dw = w * ENLARGE_RATIO / 2
        # dh = h * ENLARGE_RATIO / 2
        # nx1 = max(0, int(x1 - dw))
        # ny1 = max(0, int(y1 - dh))
        # nx2 = min(image.shape[1], int(x2 + dw))
        # ny2 = min(image.shape[0], int(y2 + dh))
        #
        # # Crop ảnh theo box mới
        # img_crop = image[ny1:ny2, nx1:nx2].copy()
        # crop_h, crop_w = img_crop.shape[:2]
        #
        # # Cập nhật lại landmark về crop
        # pts_crop = points.copy()
        # pts_crop[:, 0] = pts_crop[:, 0] - nx1
        # pts_crop[:, 1] = pts_crop[:, 1] - ny1
        #
        # # Scale landmark về image_size mới
        # scale_x = image_size / crop_w
        # scale_y = image_size / crop_h
        # pts_resize = pts_crop.copy()
        # pts_resize[:, 0] = pts_resize[:, 0] * scale_x
        # pts_resize[:, 1] = pts_resize[:, 1] * scale_y

        # Resize ảnh crop về image_size
        # img_resize = cv2.resize(img_crop, (image_size, image_size))

        # Lưu ra file
        base_name = os.path.splitext(os.path.basename(s[0]))[0]
        out_img_path = os.path.join(path_save, 'image', f"{base_name}.jpg")
        cv2.imwrite(out_img_path, image)

        out_pts_path = os.path.join(path_save, 'label', f"{base_name}.txt")
        with open(out_pts_path, 'w') as f:
            f.write(f"{len(points)}\n")
            np.savetxt(f, points, fmt="%.6f")