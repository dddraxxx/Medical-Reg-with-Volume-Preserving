import torch
import torch.nn.functional as F

def pearson_correlation(fixed, warped, soft_mask = None):
    flatten_fixed = torch.flatten(fixed, start_dim=1)
    flatten_warped = torch.flatten(warped, start_dim=1)

    mean1 = torch.mean(flatten_fixed, dim=1, keepdim=True)
    mean2 = torch.mean(flatten_warped, dim=1, keepdim=True)
    var1 = torch.mean((flatten_fixed - mean1) ** 2, dim=1, keepdim=True)
    var2 = torch.mean((flatten_warped - mean2) ** 2, dim=1, keepdim=True)

    if soft_mask is not None:
        soft_mask = torch.flatten(soft_mask, start_dim=1)
    else:
        soft_mask = 1
    cov12 = torch.mean((flatten_fixed - mean1) * (flatten_warped - mean2) * soft_mask, dim=1, keepdim=True)
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

def dice_jaccard(seg1: torch.Tensor, seg2: torch.Tensor):
    seg1 = seg1.flatten(1)
    seg2 = seg2.flatten(1)
    intersection = (seg1 * seg2).sum(1)
    union = seg1.sum(1) + seg2.sum(1)
    dice = 2 * intersection / (union + 1e-6)
    jaccard = intersection / (union - intersection + 1e-6)
    return dice, jaccard

def jacobian_det(flow, return_det=False):
    """
    flow has shape (batch, C, H, W, S)
    """
    # Compute Jacobian determinant
    batch_size, _, height, width, depth = flow.size()
    dx = flow[:, :, 1:, 1:, 1:] - flow[:, :, :-1, 1:, 1:] + flow.new_tensor([1., 0., 0.]).view(1, 3, 1, 1, 1)
    dy = flow[:, :, 1:, 1:, 1:] - flow[:, :, 1:, :-1, 1:] + flow.new_tensor([0., 1., 0.]).view(1, 3, 1, 1, 1)
    dz = flow[:, :, 1:, 1:, 1:] - flow[:, :, 1:, 1:, :-1] + flow.new_tensor([0., 0., 1.]).view(1, 3, 1, 1, 1) 
    jac = torch.stack([dx, dy, dz], dim=1).permute(0, 3, 4, 5, 1, 2)
    det = torch.det(jac)
    if return_det:
        return det
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

def sim_loss(fixed, moving, soft_mask=None):
    sim_loss = pearson_correlation(fixed, moving, soft_mask)
    return sim_loss

def masked_sim_loss(fixed, moving, mask, soft_weight=None):
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
    sim_loss = pearson_correlation(flatten_fixed, flatten_warped, soft_weight)
    return sim_loss

def reg_loss(flows):
    if len(flows[0].size()) == 4: #(N, C, H, W)
        reg_loss = sum([regularize_loss(flow) for flow in flows])
    else:
        reg_loss = sum([regularize_loss_3d(flow) for flow in flows])
    return reg_loss

def find_surf(seg, n=3):
    '''seg: (**,D,H,W)
    '''
    pads = tuple((n-1)//2 for _ in range(6))
    seg = F.pad(seg, pads, mode='constant', value=0).unfold(2, n, 1).unfold(3, n, 1).unfold(4, n, 1)
    return seg.sum(dim=(-1,-2,-3)) < n**3

def ret_surf_points(surf_p, max_num = None):
    '''surf_p: (b, c, d, h, w), binary map of whether suface
    return: (b, 3, max_num)
    '''
    sum_num = surf_p.sum((2,3,4))
    if max_num is None:
        max_num = int(min(sum_num.min(), 1e3))
    surf_pm = surf_p.new_zeros(( surf_p.size(0), surf_p.size(1), (max_num)))
    for i in range(surf_p.size(0)):
        for j in range(surf_p.size(1)):
            surf_pm[i,j] = surf_p[i, j].view(-1).nonzero()[:,0][torch.linspace(0, sum_num[i, j]-1, max_num).long()]
    return surf_pm.long().expand(-1, 3, -1)

def surf_loss(w_seg, seg1, flow):
    '''w_seg, seg1: binary map for organs
    flow: the flow field from 
    use gumble max?
    '''
    w_surf = find_surf(w_seg)
    seg1_surf = find_surf(seg1)
    b, c, d, h, w = w_surf.size()
    grid = torch.stack(torch.meshgrid(torch.arange(d), torch.arange(h), torch.arange(w),indexing='ij'), dim=0).to(w_surf.device)
    w_surf = ret_surf_points(w_surf)
    ws_points = grid.view(1,3,-1).expand(b,-1,-1).gather(2, w_surf) # b,3,n
    seg1_surf = ret_surf_points(seg1_surf)
    seg1_points = grid.view(1,3,-1).expand(b,-1,-1).gather(2, seg1_surf) # b,3,m
    # cal min distance between ws_points and seg1_points
    vect = ws_points[:, :, :, None] - seg1_points[:, :, None] # b,3,n,m
    dist = (vect**2).sum(dim=1, keepdim=True) # b,1,n,m
    a_min = dist.argmin(dim=-1, keepdim=True).expand(-1,3,-1,-1) # b,3,n,1
    ws_flow = flow.view(b,3,-1).gather(2, w_surf) # b,3,n
    surf_loss = vect.gather(-1, a_min)[...,0] - ws_flow
    # print(surf_loss.shape)
    # print(surf_loss.norm(dim=1).mean())
    return surf_loss.norm(dim=1).mean()

# if main
if __name__ == '__main__':
    import numpy as np
    np.random.seed(24)
    arr = np.random.rand(2, 3, 3)
    print(arr)
    print(ortho_loss(torch.tensor(arr).float()))