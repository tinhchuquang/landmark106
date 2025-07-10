from dataset import Landmark106Dataset
import torch.nn as nn

from torch.utils.data import DataLoader
from loss import WingLoss

from model import MobileNetV3Landmark
import torch
from config.configs import img_dir, label_dir
from config.configs import val_img_dir, val_label_dir
from config.configs import test_img_dir, test_label_dir
from config.configs import last_save_model, best_save_model

def evaluate(model, loader, criterion, wing_loss):
    # model.eval()
    total_loss = 0
    with torch.no_grad():
        for img, label in loader:
            img, label = img.cuda(), label.cuda()
            pred = model(img)
            loss = criterion(pred, label) + wing_loss(pred, label)
            total_loss += loss.item() * img.size(0)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss

def train():

    batch_size = 32
    epochs = 30
    lr = 1e-3
    img_size = 224

    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    # ])

    dataset = Landmark106Dataset(img_dir, label_dir, img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_dataset = Landmark106Dataset(val_img_dir, val_label_dir, augment=False)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)

    test_dataset = Landmark106Dataset(test_img_dir, test_label_dir, augment=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)

    model = YOLOv10Landmark(num_points=106).cuda()
    # resnet = models.resnet18(pretrained=True)
    # model = Net(resnet).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    wingloss = WingLoss(w=5.0, epsilon=2.0)

    best_val_loss = float('inf')
    for epoch in range(epochs):
        # model.train()
        total_loss = 0
        for img, label in loader:
            img, label = img.cuda(), label.cuda()
            pred = model(img)
            loss = criterion(pred, label) + wingloss(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0)
        train_loss = total_loss / len(dataset)
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.6f}")

        val_loss = evaluate(model, val_loader, criterion, wingloss)
        test_loss = evaluate(model, test_loader, criterion, wingloss)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.6f}, "
              f"Val Loss: {val_loss:.6f}, "
              f"Test Loss: {test_loss:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_save_model)

    # Save last model
    torch.save(model.state_dict(), last_save_model)

if __name__ == "__main__":
    train()