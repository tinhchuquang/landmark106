from dataset import Landmark106Dataset
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from model import MobileNetV3Landmark
import torch


def train():
    img_dir = '/media/tinhcq/data1/Training_data'
    label_dir = '/media/tinhcq/data1/Training_data/Corrected_landmark/Corrected_landmark'
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

    model = MobileNetV3Landmark(num_points=106).cuda()
    # resnet = models.resnet18(pretrained=True)
    # model = Net(resnet).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for img, label in loader:
            img, label = img.cuda(), label.cuda()
            pred = model(img)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * img.size(0)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataset):.6f}")

    torch.save(model.state_dict(), "checkpoints/106_landmark/landmark106_mobilenetv3_small.pth")

if __name__ == "__main__":
    train()