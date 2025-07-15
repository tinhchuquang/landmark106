import torch
import os
import torch.nn as nn

# from models.hrnet import get_face_alignment_net
# from config.hrnet_w18_config import config as hrnet_w18_config
from torch.utils.data import DataLoader

from models.hrnet_w18_mutilscales import HRNetW18MultiScale
from datasets.dataloader import LandmarkHeatmapDatasetMultilScale

from config.configs import img_dir, label_dir
from config.configs import val_img_dir, val_label_dir
from config.configs import test_img_dir, test_label_dir
from config.configs import last_save_model, best_save_model
from loss.head import bce_heatmap_loss

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
         for img, heatmaps56, heatmaps28, heatmaps14  in loader:
            img, heatmaps56, heatmaps28, heatmaps14 = img.cuda(), heatmaps56.cuda(), heatmaps28.cuda(), heatmaps14.cuda()
            pred56, pred28, pred14 = model(img)
            loss = bce_heatmap_loss(pred56, heatmaps56) + bce_heatmap_loss(pred14, heatmaps14) + bce_heatmap_loss(pred28, heatmaps28)
            total_loss += loss.item() * img.size(0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def train():

    batch_size = 64
    epochs = 100
    lr = 1e-3
    img_size = 224

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ])

    dataset = LandmarkHeatmapDatasetMultilScale(img_dir, label_dir, img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    val_dataset = LandmarkHeatmapDatasetMultilScale(val_img_dir, val_label_dir)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=8)

    test_dataset = LandmarkHeatmapDatasetMultilScale(test_img_dir, test_label_dir)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=8)

    # model = get_face_alignment_net(hrnet_w18_config).cuda()
    model = HRNetW18MultiScale(num_landmarks=106)
    model = nn.DataParallel(model)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # criterion = bce_heatmap_loss()

    best_val_loss = float('inf')
    for epoch in range(epochs):
        # model.train()
        total_loss = 0
        for img, heatmaps56, heatmaps28, heatmaps14  in loader:
            img, heatmaps56, heatmaps28, heatmaps14 = img.cuda(), heatmaps56.cuda(), heatmaps28.cuda(), heatmaps14.cuda()
            pred56, pred28, pred14 = model(img)
            loss = bce_heatmap_loss(pred56, heatmaps56) + bce_heatmap_loss(pred14, heatmaps14) + bce_heatmap_loss(pred28, heatmaps28)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0)
        train_loss = total_loss / len(dataset)
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.6f}")

        val_loss = evaluate(model, val_loader)
        test_loss = evaluate(model, test_loader)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Test Loss: {test_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.module.state_dict(), best_save_model)

    # Save last model
    torch.save(model.module.state_dict(), last_save_model)

if __name__ == "__main__":
    train()