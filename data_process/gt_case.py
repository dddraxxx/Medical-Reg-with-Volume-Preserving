# %%
%load_ext autoreload
# %%
%autoreload 2
from scipy.ndimage import map_coordinates
from monai.visualize import plot_2d_or_3d_image, matshow3d
import matplotlib.pyplot as plt
from utils import dict2np
import numpy as np
from find_lmk import LMK, lmks


def find_affine(pts1, pts2):
    """ find affine matrix that map pts1 to pts2
    Args:
        pts1: n x 3
        pts2: n x 3

    Returns:
        affine: 4 x 3
    """
    # find the affine matrix that map pts1 to pts2
    A = np.concatenate([pts1, np.ones((pts1.shape[0], 1))], axis=1)
    B = pts2
    affine = np.linalg.lstsq(A, B, rcond=None)[0]
    return affine


def affine(pts, aff):
    """ apply affine matrix to pts
    Args:
        pts: n x 3
        aff: 4 x 3

    Returns:
        pts: n x 3
    """
    A = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=-1)
    B = A.dot(aff)
    return B

# %% we will use the affine matrix from h5 file

# get gt flow

def affine_flow(aff, shape):
    """ get the flow of affine matrix
    Args:
        aff: 4 x 3
        shape: [x, y, z]

    Returns:
        flow: x x y x z x 3
    """
    grid = np.mgrid[:shape[0], :shape[1], :shape[2]].astype(
        np.float32).transpose(1, 2, 3, 0)
    aff_grid = affine(grid.reshape(-1, 3),
                      aff).reshape(shape[0], shape[1], shape[2], 3)
    flow = aff_grid - grid
    return flow

def read_pred_flow(path):
    """ read the predicted flow
    Args:
        path: path to the predicted flow

    Returns:
        id1, id2: the id of the fixed and moving image
        flow: x x y x z x 3
    """
    import pickle as pkl
    pred = pkl.load(open(path, 'rb'))
    l = [[pred['id1'][idx],
          pred['id2'][idx],
          pred['flow'][idx].permute(1, 2, 3, 0).numpy(),
          pred['warped'][idx][0].cpu().numpy(),
          pred['wseg2'][idx][0].cpu().numpy(),]
         for idx in range(len(pred['id1']))]
    return np.array(l)

#%%
p1 = lmks[1][0].lps_lmks
p1_np = np.array([p['position']
                 for p in sorted(p1, key=lambda x: int(x['id']))])
p2 = lmks[1][1].lps_lmks
p2_np = np.array([p['position']
                 for p in sorted(p2, key=lambda x: int(x['id']))])

p1_h5 = lmks[1][0].h5_coord
p1_h5_np = dict2np(p1_h5)
p2_h5 = lmks[1][1].h5_coord
p2_h5_np = dict2np(p2_h5)
print(p1_h5_np, '\n', p2_h5_np)

# the flow field that warp p1 to p2
gt_aff = find_affine(p1_h5_np, p2_h5_np)
# %%
error = p2_h5_np - affine(p1_h5_np, gt_aff)
e = np.linalg.norm(error, axis=-1)
print(e.max(), e.mean())  # pretty accurate
gt_flow = affine_flow(gt_aff, [128, 128, 128])

# %%
# get predicted flow
normal_path = '/home/hynx/regis/recursive_cascaded_networks/eval/eval_pkls/Mar02-050938_1_VTNx3_normal____2_lm10.pkl'
vp_path = '/home/hynx/regis/recursive_cascaded_networks/eval/eval_pkls/Mar03-014518_br1_VTNx3_adaptive_softthr1.5sigmbnd0.5st1_vp0.1stdynamic_2_lm10.pkl'

normal = read_pred_flow(normal_path)
vp = read_pred_flow(vp_path)

#%%
# fixed: p1-1, moving: p1-2
nor_flow = normal[6][2]
vp_flow = vp[6][2]

# %% calculate error for all the flow
def get_err(pred, gt, seg=None):
    err = np.linalg.norm(pred - gt, axis=-1)
    if seg is not None:
        err = err * (seg>0)
    return err
def cal_err(pred, gt, seg=None):
    err = np.linalg.norm(pred - gt, axis=-1)
    dct = {}
    if seg is not None:
        dct['organ'] = {}
        organ_err = err * (seg>0.5)
        dct['organ']['max'] = organ_err.max()
        dct['organ']['mean'] = organ_err.sum() / (seg>0.5).sum()
        dct['organ']['std'] = organ_err.std()

        dct['tumor'] = {}
        tumor_err = err * (seg>1.5)
        dct['tumor']['max'] = tumor_err.max()
        dct['tumor']['mean'] = tumor_err.sum() / (seg>1.5).sum()
        dct['tumor']['std'] = tumor_err.std()

        dct['whole'] = {}
        dct['whole']['max'] = err.max()
        dct['whole']['mean'] = err.mean()
        dct['whole']['std'] = err.std()

        return dct
    return {
        'max': err.max(),
        'mean': err.mean(),
        'std': err.std(),
    }

from pprint import pprint
print('normal\n')
pprint(cal_err(nor_flow, gt_flow, normal[6][4]))
print('vp\n')
pprint(cal_err(vp_flow, gt_flow, vp[6][4]))

#%%
nerr = get_err(nor_flow, gt_flow, normal[6][4])
verr = get_err(vp_flow, gt_flow, vp[6][4])
n_verr = (nerr-verr)
n_verr = np.clip(n_verr, -1, 1)
#%%
nor_warped = normal[6][3]
nor_seg = normal[6][4]
matshow3d(nor_seg, every_n=5, cmap='gray')

# %% visualizin the flow error
fig = plt.figure(figsize=(10, 10))
matshow3d(nor_warped, every_n=5, cmap='gray', fig=fig)
nor_err = ((nor_flow - gt_flow) ** 2).sum(-1)
matshow3d(n_verr, every_n=5, fig=fig, alpha=0.2, cmap='coolwarm')
# color bar
fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=fig.axes)

#%%
fig = plt.figure(figsize=(10, 10))
vp_warped = vp[5][3]
# matshow3d(vp_warped, every_n=5, cmap='gray', fig=fig)
vp_err = ((vp_flow - gt_flow) ** 2).sum(-1)
matshow3d(vp_err, every_n=5, fig=fig, alpha=0.5, cmap='jet')
fig.colorbar(plt.cm.ScalarMappable(cmap='jet'), ax=fig.axes)







# %% calculate 18 landmaroks error
def warp_lmk(flow, pts):
    """ warp the landmarks
    Args:
        flow: x x y x z x 3
        pts: n x 3

    Returns:
        warped_pts: n x 3
    """
    grid = np.mgrid[:flow.shape[0], :flow.shape[1],
                    :flow.shape[2]].astype(np.float32).transpose(1, 2, 3, 0)
    after_grid = grid + flow
    warped_pts = []
    for i in range(3):
        warped_pts.append(map_coordinates(after_grid[..., i], pts.T, order=1))
    return np.array(warped_pts).T


nor_warped_lmk = warp_lmk(nor_flow, p1_h5_np)
vp_warped_lmk = warp_lmk(vp_flow, p1_h5_np)
gt_warped_lmk = warp_lmk(gt_flow, p1_h5_np)

err_nor = np.linalg.norm(nor_warped_lmk - p2_h5_np, axis=-1)
err_vp = np.linalg.norm(vp_warped_lmk - p2_h5_np, axis=-1)
err_gt = np.linalg.norm(gt_warped_lmk - p2_h5_np, axis=-1)
print(err_nor.mean(), err_nor.max())
print(err_vp.mean(), err_vp.max())
print(err_gt.mean(), err_gt.max())

# draw err plot
plt.plot(err_nor, label='normal')
plt.plot(err_vp, label='vp')
# show x axis ticks 1-18
plt.xticks(np.arange(18), np.arange(1, 19))
# grid
plt.grid()
# legend
plt.legend()
