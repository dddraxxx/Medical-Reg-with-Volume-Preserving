import torch


def pearson_correlation(fixed, warped):
    flatten_fixed = torch.flatten(fixed, start_dim=1)
    flatten_warped = torch.flatten(warped, start_dim=1)

    mean1 = torch.mean(flatten_fixed, dim=1, keepdim=True)
    mean2 = torch.mean(flatten_warped, dim=1, keepdim=True)
    var1 = torch.mean((flatten_fixed - mean1) ** 2, dim=1, keepdim=True)
    var2 = torch.mean((flatten_warped - mean2) ** 2, dim=1, keepdim=True)

    cov12 = torch.mean((flatten_fixed - mean1) * (flatten_warped - mean2), dim=1, keepdim=True)
    eps = 1e-6
    pearson_r = cov12 / torch.sqrt((var1 + eps) * (var2 + eps))

    raw_loss = 1 - pearson_r

    # not mean in official implementation
    return raw_loss.sum()


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

    # follows official implementation
    d = torch.mean(dx**2, dim=(1,2,3,4)) + torch.mean(dy**2, dim=(1,2,3,4)) + torch.mean(dz**2, dim=(1,2,3,4))
    return d.sum()/2.0


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

def ortho_loss(A: torch.Tensor):
    eps = 1e-5
    epsI = eps * torch.eye(3).to(A.device)[None]
    C = A.transpose(-1,-2)@A + epsI
    def elem_sym_polys_of_eigenvalues(M):
        M = M.permute(1,2,0)
        sigma1 = M[0,0] + M[1,1] + M[2,2]
        sigma2 = M[0,0]*M[1,1] + M[0,0]*M[2,2] + M[1,1]*M[2,2] - M[0,1]**2 - M[0,2]**2 - M[1,2]**2
        sigma3 = M[0,0]*M[1,1]*M[2,2] + 2*M[0,1]*M[0,2]*M[1,2] - M[0,0]*M[1,2]**2 - M[1,1]*M[0,2]**2 - M[2,2]*M[0,1]**2
        return sigma1, sigma2, sigma3
    s1, s2, s3 = elem_sym_polys_of_eigenvalues(C)
    # ortho_loss = s1 + s2/s3 - 6
    # original formula
    ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
    return ortho_loss.sum()

def det_loss(A):
    # calculate the determinant of the affine matrix
    det = torch.det(A)
    # l2 loss
    return torch.sum(0.5 * (det - 1) ** 2)

def sim_loss(fixed, moving):
    sim_loss = pearson_correlation(fixed, moving)
    return sim_loss

def masked_sim_loss(fixed, moving, mask):
    flatten_fixed = torch.flatten(fixed, start_dim=1)
    flatten_warped = torch.flatten(moving, start_dim=1)
    flatten_mask = torch.flatten(mask, start_dim=1)
    # get masked mean: non-masked sum / non-masked count
    flatten_fixed[flatten_mask] = 0
    flatten_warped[flatten_mask] = 0
    fixed_mean = flatten_fixed.sum(1) / (flatten_mask.shape[1] - flatten_mask.sum(1))
    warped_mean = flatten_warped.sum(1) / (flatten_mask.shape[1] - flatten_mask.sum(1))
    # replace masked values with masked mean
    flatten_fixed = torch.where(flatten_mask, fixed_mean[:, None], flatten_fixed)
    flatten_warped = torch.where(flatten_mask, warped_mean[:, None], flatten_warped)
    # calculate pearson correlation
    sim_loss = pearson_correlation(flatten_fixed, flatten_warped)
    return sim_loss

def reg_loss(flows):
    if len(flows[0].size()) == 4: #(N, C, H, W)
        reg_loss = sum([regularize_loss(flow) for flow in flows])
    else:
        reg_loss = sum([regularize_loss_3d(flow) for flow in flows])
    return reg_loss

# if main
if __name__ == '__main__':
    import numpy as np
    np.random.seed(24)
    arr = np.random.rand(2, 3, 3)
    print(arr)
    print(ortho_loss(torch.tensor(arr).float()))

