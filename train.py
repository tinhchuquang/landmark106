import torch
import os
import torch.nn as nn

from torch.utils.data import DataLoader
import torchvision.transforms as T

from models.yolo import YOLOv10_Landmark_Heatmap
from datasets.datasets_heatmap import WLFWDatasets
from loss.head import L2_98landmark_loss

os.environ["CUDA_VISIBLE_DEVICES"]="5"

def evaluate(model, loader, criterion):
    # model.eval()
    total_loss = 0
    with torch.no_grad():
        for img, label, _, _ in loader:
            img, label = img.cuda(), label.cuda()
            pred = model(img)
            loss = criterion(pred, label)
            total_loss += loss.item() * img.size(0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def train():

    batch_size = 32
    epochs = 50
    lr = 1e-3
    img_size = 224


    root_dir = '/data2/tinhcq/WFLW'
    imageDirs = '/data2/tinhcq/WFLW/WFLW_images'
    Mirror_file = './data/Mirror98.txt'
    landmarkDirs = ['/data2/tinhcq/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt',
                    '/data2/tinhcq/WFLW/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt']

    transforms = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    train_dataset = WLFWDatasets(imageDirs, Mirror_file, landmarkDirs[1], is_train=True, transforms=transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    test_dataset = WLFWDatasets(imageDirs, Mirror_file, landmarkDirs[0], is_train=False, transforms=transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    # model = get_face_alignment_net(hrnet_w18_config).cuda()
    model = YOLOv10_Landmark_Heatmap(num_landmarks=98, heatmap_size=56)
    model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = L2_98landmark_loss()

    best_test_loss = float('inf')
    for epoch in range(epochs):
        # model.train()
        total_loss = 0
        for img, label, _, _ in train_loader:
            img, label = img.cuda(), label.cuda()
            pred = model(img)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0)
        train_loss = total_loss / len(train_dataset)
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.6f}")

        # val_loss = evaluate(model, val_loader)
        test_loss = evaluate(model, test_loader, criterion)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.6f}, "
            #   f"Val Loss: {val_loss:.6f}, "
              f"Test Loss: {test_loss:.6f}")

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.module.state_dict(), 'checkpoints/yolo/best_model_heatmaps.pth')

    # Save last model
    torch.save(model.module.state_dict(), 'checkpoints/yolo/last_model_heatmaps.pth')

if __name__ == "__main__":
    train()