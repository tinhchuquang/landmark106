import torch
import cv2
import numpy as np
from model import MobileNetV3Landmark  # hoặc import class model của bạn
import matplotlib.pyplot as plt
import torchvision.models as models

# ==== THÔNG SỐ ====
num_points = 106
input_size = 224
img_path = '/media/tinhcq/data1/Training_data/AFW/picture/AFW_45092961_1_4.jpg'   # Đổi thành đường dẫn ảnh của bạn
model_weight = 'landmark106_mobilenetv3_small.pth'  # Nếu có weight đã train

# ==== Tiền xử lý ảnh ====
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h0, w0 = img.shape[:2]
img_resized = cv2.resize(img_rgb, (input_size, input_size))
img_norm = img_resized.astype(np.float32) / 255.0
img_norm = (img_norm - 0.5) / 0.5  # Nếu bạn training với mean=0.5 std=0.5
img_tensor = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0)  # shape (1,3,224,224)

# ==== Load model ====
# resnet = models.resnet18(pretrained=False)
# model = Net(resnet)
model = MobileNetV3Landmark(num_points=num_points)
# Nếu có weight:
model.load_state_dict(torch.load(model_weight, map_location='cpu'))
model.eval()

# ==== Inference ====
with torch.no_grad():
    pred = model(img_tensor)  # shape [1, 212]
    pred = pred.view(num_points, 2).numpy()
# Đưa về pixel trên ảnh gốc
pred[:,0] = pred[:,0] * w0    # nếu label/gt cũng chuẩn hóa về [0,1]
pred[:,1] = pred[:,1] * h0

# ==== Vẽ landmark lên ảnh gốc ====
img_vis = img_rgb.copy()
for (x, y) in pred:
    cv2.circle(img_vis, (int(x), int(y)), 2, (255, 0, 0), -1)

# ==== Lưu và show kết quả ====
cv2.imwrite('output_landmark.jpg', cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR))
plt.imshow(img_vis)
plt.title('Predicted 106 Landmark')
plt.axis('off')
plt.show()
print("Landmark predicted và lưu ra output_landmark.jpg")