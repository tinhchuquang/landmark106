from dataset import Landmark106Dataset
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from model import LandmarkModel
import torch


def train():
    img_dir = "dataset/images"
    label_dir = "dataset/labels"
    batch_size = 32
    epochs = 30
    lr = 1e-3
    img_size = 224

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    dataset = Landmark106Dataset(img_dir, label_dir, img_size, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = LandmarkModel(num_points=106).cuda()
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

    torch.save(model.state_dict(), "landmark106_mobilenetv2.pth")

if __name__ == "__main__":
    train()