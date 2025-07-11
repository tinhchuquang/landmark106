import torch
import torch.nn.functional as F
import torch.nn as nn


def focal_loss(pred, target, alpha=2.0, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.exp(-bce)
    loss = (alpha * (1-pt)**gamma * bce).mean()
    return loss


class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0, reduction='mean'):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = w - w * torch.log(torch.tensor(1 + w / epsilon))
        self.reduction = reduction

    def forward(self, pred, target):
        """
        pred:   (B, K, H, W) - heatmap dự đoán (float)
        target: (B, K, H, W) - groundtruth heatmap (float)
        """
        x = pred - target
        abs_x = torch.abs(x)
        # Phần tính loss: nếu abs_x < w thì dùng log, ngược lại dùng tuyến tính
        loss = torch.where(
            abs_x < self.w,
            self.w * torch.log(1 + abs_x / self.epsilon),
            abs_x - self.C
        )
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss