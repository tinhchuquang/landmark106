import cv2

def crop_center(image, center, size=450):
    h, w = image.shape[:2]
    cx, cy = center
    x1 = int(cx - size/2)
    y1 = int(cy - size/2)
    x2 = x1 + size
    y2 = y1 + size
    pad_x1 = max(0, -x1)
    pad_y1 = max(0, -y1)
    pad_x2 = max(0, x2-w)
    pad_y2 = max(0, y2-h)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    cropped = image[y1:y2, x1:x2]
    if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
        cropped = cv2.copyMakeBorder(cropped, pad_y1, pad_y2, pad_x1, pad_x2, borderType=cv2.BORDER_CONSTANT, value=0)
    return cropped, (x1, y1)