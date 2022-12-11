#%%
from pprint import pprint
import numpy as np
import pickle as pkl
import numpy as np
import os
from pathlib import Path as pa
import sys

sys.path.append('..')
sys.path.append('.')
from networks.layers import SpatialTransformer
from utils import cal_rev_flow, draw_seg_on_vol, show_img, visualize_3d, combo_imgs, cal_single_warped
import torch
import torch.nn.functional as F
from vis_3d_ffd import FFD_GT


def eP2P(flow, gt, mask):
    eP2P = abs(flow) - abs(gt)
    # no tumor in
    eP2P = eP2P[~mask]
    # print(eP2P.shape)
    err = ((eP2P**2).sum(axis=-1)).mean()
    return err

def cal_warped_seg(agg_flows, seg2):
    st = SpatialTransformer(seg2[0,0].shape)
    w_segs = st(seg2, agg_flows, mode='nearest')
    return w_segs

def single_jacobian_det(flow):
    """
    flow has shape (batch, C, H, W, S)
    """
    flow = flow[None]
    # Compute Jacobian determinant
    batch_size, _, height, width, depth = flow.shape
    dx = flow[:, :, 1:, 1:, 1:] - flow[:, :, :-1, 1:, 1:] + np.array([1,0,0])[:,None,None,None]
    dy = flow[:, :, 1:, 1:, 1:] - flow[:, :, 1:, :-1, 1:] + np.array([0,1,0])[:,None,None,None]
    dz = flow[:, :, 1:, 1:, 1:] - flow[:, :, 1:, 1:, :-1] + np.array([0,0,1])[:,None,None,None]
    jac = np.stack([dx, dy, dz], axis=1).transpose(0, 3, 4, 5, 1, 2)
    det = np.linalg.det(jac)
    return det[0]

def single_grid_volumes(flow):
    """
    flow has shape (batch, C, H, W, S)
    """
    flow = flow[None]
    # Compute Jacobian determinant
    batch_size, _, height, width, depth = flow.shape
    dx = flow[:, :, 1:, 1:, 1:] - flow[:, :, :-1, 1:, 1:] + np.array([1,0,0])[:,None,None,None]
    dy = flow[:, :, 1:, 1:, 1:] - flow[:, :, 1:, :-1, 1:] + np.array([0,1,0])[:,None,None,None]
    dz = flow[:, :, 1:, 1:, 1:] - flow[:, :, 1:, 1:, :-1] + np.array([0,0,1])[:,None,None,None]
    dxy = flow[:, :, 1:, 1:, 1:] - flow[:, :, :-1, :-1, 1:] + np.array([1,1,0])[:,None,None,None]
    dxz = flow[:, :, 1:, 1:, 1:] - flow[:, :, :-1, 1:, :-1] + np.array([1,0,1])[:,None,None,None]
    dyz = flow[:, :, 1:, 1:, 1:] - flow[:, :, 1:, :-1, :-1] + np.array([0,1,1])[:,None,None,None]
    dxyz = flow[:, :, 1:, 1:, 1:] - flow[:, :, :-1, :-1, :-1] + np.array([1,1,1])[:,None,None,None]
    triangle_grid = np.stack([[0*dx, dx, dy, dz],
                                [dxy, dx, dy, dz],
                                [dyz, dxy, dy, dz],
                                [dxyz, dxy, dyz, dz],
                                [dz, dy, dxy, dxyz],
                                [dz, dx, dxz, dxyz]], 
                                axis=1).transpose(2, 4,5,6,1,0,3)
    vol_grid = triangle_grid[..., 1:,:]-triangle_grid[..., :1,:]
    det = np.linalg.det(vol_grid)
    return det[0].sum(axis=-1)/6


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #%%
    fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Dec06_012136_normal-vtn_lits_d_.pkl'
    fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Dec06_012136_normal-vtn_mini-val_.pkl'
    mode = 'normal'
    # fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Oct03_210121_masked_lits_d_.pkl'
    # mode = 'mask'
    # fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Dec06_043431_msk-ks-vtn_lits_d_.pkl'
    # mode = 'mask_ks'
    # below: bad performance
    # fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Oct11_183942_soft-m_lits_d_.pkl'
    # mode = 'soft_mask'
    # fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Oct16_164126_normal_o.01_lits_d_.pkl'
    # mode = 'normal_o.01'
    # fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Nov26_030214_normal-vxm_lits_d_.pkl'
    # mode = 'vxm'
    # fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Dec06_050505_hard-ks-vtn_lits_d_.pkl'
    # mode = 'hard-ks'

    dct = pkl.load(open(fname, 'rb'))
    print(dct.keys())
    print(dct['dice_liver'])
    #%%
    idx = 0
    errs = []

    h5_name = '/home/hynx/regis/recursive-cascaded-networks/datasets/lits_deform_fL.h5'
    reader = FFD_GT(h5_name)

    save_im = False
    # for idx in range(len(dct['id1'])):
    for idx in [0]:
        agg_flow = dct['agg_flows'][idx][-1].permute(1,2,3,0).numpy()
        
        id1 = dct['id1'][idx][7:]
        id2 = dct['id2'][idx]
        print('analyzing', id1, id2)
        reader.set_to(id1, id2);
        #%%
        gt_flow = reader.return_flow()
        seg2 = reader.seg2
        seg1 = reader.seg1
        masks = [seg2>1.5]
        thres = np.percentile(reader.p_disp, [90, 95, 99])
        masks = masks + [np.logical_or(seg2>1.5, m) for m in reader.get_masks(thres)]
        p_masks = [m for m in reader.get_masks(thres)]
        img1 = reader.img1
        img2 = reader.img2
        normal_w_img2 = cal_single_warped(agg_flow.transpose(3,0,1,2), img2)
        gt_w_img2 = cal_single_warped(gt_flow.transpose(3,0,1,2), img2)
        #%%
        %run -i '/home/hynx/regis/recursive-cascaded-networks/networks/layers.py'
        def cal_rev_flow1(flow):
            flow = torch.tensor(flow).permute(-1,0,1,2).cuda()[None]
            size = flow.shape[2:]
            rs = RevSpatialTransformer(size)
            rs.cuda()
            return rs(flow)[0].permute(1,2,3,0).cpu().numpy()
        ragg_flow = cal_rev_flow1(agg_flow)
        areas = abs(single_jacobian_det(ragg_flow.transpose(3,0,1,2)))
        # neg_flow = dct['rev_flow'][idx].permute(1,2,3,0).numpy()
        # areas = abs(single_jacobian_det(neg_flow.transpose(3,0,1,2)))
        areas = torch.tensor(areas)
        areas = 1/areas
        # get nxn neighbors of each voxel and avg areas
        def neighbor_avg(arr, n):
            arr = F.pad(torch.as_tensor(arr), pad=(n//2, n//2, n//2, n//2, n//2, n//2), mode='constant', value=0).unfold(0, n, 1).unfold(1, n, 1).unfold(2, n, 1)
            return arr.mean(axis=(-1,-2,-3))
        nb_areas = neighbor_avg(areas, 10)
        #%%
        values = np.percentile(nb_areas, [80, 85, 90, 95, 99])
        # values = np.percentile(1/nb_areas, [80, 85, 90, 95, 99])
        q_areas = np.digitize(nb_areas, values)
        from tools.utils import plt_img3d_axes
        sum_masks = np.sum(masks, axis=0)
        sum_p_masks = np.sum(p_masks, axis=0)
        import matplotlib
        def func_a(ax, i):
            p = ax.imshow(sum_masks[i], cmap='RdPu', alpha=.7, norm=matplotlib.colors.Normalize(vmin=0, vmax=4))
            # p = ax.imshow(sum_p_masks[i], cmap='RdPu', alpha=.6, norm=matplotlib.colors.Normalize(vmin=0, vmax=3))
            # plt.colorbar(p, ax=ax, ticks=list(range(len(masks))))

            # d = ax.contour(nb_areas[i], levels=values[:], linewidths=1, alpha=.7, colors=['k','r', 'g','b', 'y'])
            # ax.clabel(d, inline=True, fontsize=5)
            # plt.colorbar(d, ax=ax, ticks=values[-3:])

            p = ax.imshow((nb_areas.clip(0,2)[i]), cmap='gray', alpha=.7)
            # plt.colorbar(p, ax=ax)
            # ax.imshow(reader.img2[i], cmap='gray', alpha=.5)
            # ax.imshow(normal_w_img2[i], cmap='gray', alpha=.9)
            # ax.imshow(gt_w_img2[i], cmap='gray', alpha=.5)
            ax.axis('off')
        plt_img3d_axes(areas, func_a);
        #%% check r_ffd (larger disp than p_ffd)
        # r_ffd = reader.r_ffd
        # def ffd_flow(ffd, size):
        #     pt = np.mgrid[0:size[0], 0:size[1], 0:size[2]].astype(np.float32).reshape(3, -1).T
        #     flow = ffd(pt)-pt
        #     flow = flow.T.reshape(3,*size)
        #     return flow
        # r_ffd_flow = ffd_flow(r_ffd, seg2.shape)
        # p_ffd_flow = ffd_flow(reader.p_ffd, seg2.shape)
        # #%%
        # # np.histogram(r_ffd_flow)
        # show_img(np.linalg.norm(reader.gt_flow, axis=-1))
        # show_img(np.linalg.norm(p_ffd_flow, axis=0))
        # np.histogram(np.linalg.norm(r_ffd_flow, axis=0)),np.histogram(np.linalg.norm(p_ffd_flow, axis=0))

        #%% calculate organ/tumor area change
        w_segs= cal_warped_seg(dct['agg_flows'][idx], torch.tensor(seg2)[None,None].repeat(4,1,1,1,1).float())
        for i, s in enumerate([seg2, *w_segs, seg1]):
            print('{}th layer| organ size {}| tumor size {}'.format(i, (s>0.5).sum(), (s>1.5).sum()))

        err = [eP2P(agg_flow, gt_flow, m) for m in masks]
        print('{} to {} flow end to end error:'.format(id1, id2), err)
        errs.append(err)

        #%% save image
        if save_im:
            img_dir = pa('../images/{}/{}_{}'.format(mode, id1, id2))
            img_dir.mkdir(parents=True, exist_ok=True)
            # draw tumor seg to img2
            s_img2 = draw_seg_on_vol(img2[None], (seg2==2)[None], colors=[(250,0,0)])
            # save warped
            warped = cal_single_warped(dct['agg_flows'][idx][-1], torch.tensor(img2).float())
            s_warped = draw_seg_on_vol(warped[None], (w_segs[-1]==2), colors=[(250,0,0)])
            combo_imgs(img1, s_img2, s_warped).save(img_dir/'deform.png')
            print(img_dir/'deform.png')
            # save thresholded masks
            combo_imgs(*[m.astype(int)+(seg2>0)+(seg2==2) for m in masks]).save(img_dir/'masks.png')
            print(img_dir/'masks.png')
            #%% save eP2P error
            from tools.utils import plt_img3d_axes
            p2p_err = agg_flow-gt_flow
            p2p_err = np.linalg.norm(p2p_err, axis=-1)
            p2p_err[masks[2]]=0
            def func(ax, i):
                ax.imshow(p2p_err[i], cmap='jet', alpha=.5)
                # d = ax.contour(p2p_err[i], levels=5, linewidths=.8)
                # ax.clabel(d, inline=True, fontsize=5)
                ax.imshow(img2[i], cmap='gray', alpha=.5)
            plt_img3d_axes(p2p_err, func).savefig(img_dir/'p2p_err.png')
            # show_img(p2p_err).save(img_dir/'p2p_err.png')
            print(img_dir/'p2p_err.png')
            plt_img3d_axes(areas, func_a).savefig(img_dir/'areas.png')
            print(img_dir/'areas.png')
        #%%
        print()
    print(errs)
    print(mode, 'mean pnp error:', np.mean(errs, axis=0), 'std:', np.std(errs, axis=0))

    #%% close reader
    reader.close()