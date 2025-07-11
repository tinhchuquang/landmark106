import torch


def soft_argmax(heatmap):
    # heatmap: [N, H, W] hoặc [B, N, H, W], torch tensor
    if heatmap.ndim == 4:
        B, N, H, W = heatmap.shape
        heatmap = heatmap.reshape(B, N, -1)
        heatmap = torch.softmax(heatmap, dim=-1)
        coords = heatmap.argmax(dim=-1)
        xs = (coords % W).float()
        ys = (coords // W).float()
        return torch.stack([xs, ys], dim=-1)  # [B, N, 2]
    elif heatmap.ndim == 3:
        N, H, W = heatmap.shape
        heatmap = heatmap.reshape(N, -1)
        heatmap = torch.softmax(heatmap, dim=-1)
        coords = heatmap.argmax(dim=-1)
        xs = (coords % W).float()
        ys = (coords // W).float()
        return torch.stack([xs, ys], dim=-1)
