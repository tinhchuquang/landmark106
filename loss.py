import torch
import torch.nn as nn


class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0):
        super(WingLoss, self).__init__()
        self.w = w
        self.epsilon = epsilon

    def forward(self, pred, target):
        # pred, target: [B, 212] (hoặc [B, 106, 2])
        diff = pred - target
        abs_diff = torch.abs(diff)
        flag = abs_diff < self.w
        C = self.w - self.w * torch.log(torch.tensor(1.0 + self.w / self.epsilon, device=abs_diff.device))
        loss = torch.where(
            flag,
            self.w * torch.log(1.0 + abs_diff / self.epsilon),
            abs_diff - C
        )
        return loss.mean()
