import torch


def pearson_correlation(fixed, warped):
    flatten_fixed = torch.flatten(fixed, start_dim=1)
    flatten_warped = torch.flatten(warped, start_dim=1)

    mean1 = torch.mean(flatten_fixed)
    mean2 = torch.mean(flatten_warped)
    var1 = torch.mean((flatten_fixed - mean1) ** 2)
    var2 = torch.mean((flatten_warped - mean2) ** 2)

    cov12 = torch.mean((flatten_fixed - mean1) * (flatten_warped - mean2))
    eps = 1e-6
    pearson_r = cov12 / torch.sqrt((var1 + eps) * (var2 + eps))

    raw_loss = 1 - pearson_r

    return raw_loss


def regularize_loss(flow):
    """
    flow has shape (batch, 2, 521, 512)
    """
    dx = (flow[..., 1:, :] - flow[..., :-1, :]) ** 2
    dy = (flow[..., 1:] - flow[..., :-1]) ** 2

    d = torch.mean(dx) + torch.mean(dy)

    return d / 2.0


def regularize_loss_3d(flow):
    """
    flow has shape (batch, 3, 512, 521, 512)
    """
    dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]

    d = torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)

    return d / 3.0


def dice_loss(fixed_mask, warped):
    """
    Dice similirity loss
    """

    epsilon = 1e-6

    flat_mask = torch.flatten(fixed_mask, start_dim=1)
    flat_warp = torch.abs(torch.flatten(warped, start_dim=1))
    intersection = torch.sum(flat_mask * flat_warp)
    denominator = torch.sum(flat_mask) + torch.sum(flat_warp) + epsilon
    dice = (2.0 * intersection + epsilon) / denominator

    return 1 - dice

def score_metrics(seg1: torch.Tensor, seg2: torch.Tensor):
    seg1 = seg1.flatten(1)
    seg2 = seg2.flatten(1)
    intersection = (seg1 * seg2).sum(1)
    union = seg1.sum(1) + seg2.sum(1)
    dice = 2 * intersection / (union + 1e-6)
    jaccard = intersection / (union - intersection + 1e-6)
    return dice, jaccard

def jacobian_det(flow):
    """
    flow has shape (batch, C, H, W, S)
    """
    # Compute Jacobian determinant
    batch_size, _, height, width, depth = flow.size()
    dx = flow[:, :, 1:, 1:, 1:] - flow[:, :, :-1, 1:, 1:] + 1
    dy = flow[:, :, 1:, 1:, 1:] - flow[:, :, 1:, :-1, 1:] + 1
    dz = flow[:, :, 1:, 1:, 1:] - flow[:, :, 1:, 1:, :-1] + 1
    jac = torch.stack([dx, dy, dz], dim=1).permute(0, 3, 4, 5, 1, 2)
    det = torch.det(jac)
    # return variance of det
    return torch.var(det, dim=[1, 2, 3])

def total_loss(fixed, moving, flows):
    sim_loss = pearson_correlation(fixed, moving)
    # Regularize all flows
    if len(fixed.size()) == 4: #(N, C, H, W)
        reg_loss = sum([regularize_loss(flow) for flow in flows])
    else:
        reg_loss = sum([regularize_loss_3d(flow) for flow in flows])
    return sim_loss + reg_loss, reg_loss

