import cv2
import numpy as np
import os
from scrfd import SCRFD

def load_grand_challenge():
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
    return samples

def load_lapa():
    label_dir = '/media/tinhcq/data1/Training_data/LaPa/test/landmarks'
    img_dir = '/media/tinhcq/data1/Training_data/LaPa/test/images'
    samples = []
    for fname in os.listdir(label_dir):
        label_path = os.path.join(label_dir, fname)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        num_points = len(lines) - 1
        if num_points != 106:
            continue
        image_name = fname.replace('.txt', '.jpg')
        img_path = os.path.join(img_dir, image_name)
        if os.path.exists(img_path):
            samples.append((img_path, label_path))
    return samples

if __name__ == '__main__':
    detector = SCRFD(model_file='checkpoints/scrfd/scrfd_500m_bnkps.onnx')
    detector.prepare(-1)

    # samples = load_grand_challenge()
    samples = load_lapa()

    path_save = '/media/tinhcq/data1/Training_data/Lapa_Heatmap/test'
    os.makedirs(os.path.join(path_save, 'images'), exist_ok=True)
    os.makedirs(os.path.join(path_save, 'landmarks'), exist_ok=True)



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
        bboxes, kpss = detector.detect(image, 0.5, input_size=(640, 640))
        if len(bboxes) == 0:
            print(f"Không detect được face: {s[0]}")
            continue

        # === Tính center landmark ===
        box = bboxes[0][:4]  # lấy bbox đầu tiên (hoặc bạn chọn bbox có score lớn nhất)
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        center = np.array([center_x, center_y], dtype=np.float32)

        # Lưu ra file
        base_name = os.path.splitext(os.path.basename(s[0]))[0]
        out_img_path = os.path.join(path_save, 'images', f"{base_name}.jpg")
        cv2.imwrite(out_img_path, image)

        out_pts_path = os.path.join(path_save, 'landmarks', f"{base_name}.txt")
        with open(out_pts_path, 'w') as f:
            f.write(f"{len(points)}\n")
            f.write(f"{center[0]:.6f} {center[1]:.6f}\n")
            np.savetxt(f, points, fmt="%.6f")