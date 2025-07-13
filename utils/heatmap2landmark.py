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


def heatmaps_to_landmarks(heatmaps, img_size=224):
    """
    heatmaps: [B, num_points, H, W] (torch.Tensor)
    return: [B, num_points, 2] (x, y) trên ảnh size img_size x img_size
    """
    B, num_points, h, w = heatmaps.shape
    heatmaps_flat = heatmaps.view(B, num_points, -1)             # [B, 106, H*W]
    max_idx = heatmaps_flat.argmax(dim=2)                        # [B, 106]
    y = (max_idx // w).float()
    x = (max_idx % w).float()
    x_img = x / w * img_size
    y_img = y / h * img_size
    coords = torch.stack([x_img, y_img], dim=2)                  # [B, 106, 2]
    return coords
