import numpy as np

image_size = 224
num_points = 106
mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
std  = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
ENLARGE_RATIO = 0.4
img_dir = ['/data2/tinhcq/Training_data/Lapa_Heatmap/train/images', '/data2/tinhcq/Training_data/Data/images']
label_dir = ['/data2/tinhcq/Training_data/Lapa_Heatmap/train/landmarks', '/data2/tinhcq/Training_data/Data/landmarks']
val_img_dir = ['/data2/tinhcq/Training_data/Lapa_Heatmap/val/images']
val_label_dir = ['/data2/tinhcq/Training_data/Lapa_Heatmap/val/landmarks']
test_img_dir = ['/data2/tinhcq/Training_data/Lapa_Heatmap/test/images']
test_label_dir = ['/data2/tinhcq/Training_data/Lapa_Heatmap/test/landmarks']
best_save_model = 'checkpoints/106_landmark/best_landmark106_hrnetw18_multilscales.pth'
last_save_model = 'checkpoints/106_landmark/landmark106_hrnetw18_multilscales.pth'

dst_pts = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

dst_pts_128 = np.array([
    [54.7061, 64.0000],
    [104.2941, 64.0000],
    [79.5001, 89.0000],
    [59.0001, 113.0000],
    [100.0001, 113.0000],
], dtype=np.float32)