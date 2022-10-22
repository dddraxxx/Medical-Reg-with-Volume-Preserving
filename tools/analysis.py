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
from networks.spatial_transformer import SpatialTransform
from utils import draw_seg_on_vol, show_img, visualize_3d, combo_imgs
import torch
from thres_ffd import FFD_GT, cal_single_warped


def eP2P(flow, gt, mask):
    eP2P = abs(flow) - abs(gt)
    # no tumor in
    eP2P = eP2P[~mask]
    # print(eP2P.shape)
    err = ((eP2P**2).sum(axis=-1)).mean()
    return err

def cal_warped_seg(agg_flows, seg2):
    st = SpatialTransform(seg2[0,0].shape)
    w_segs = st(seg2, agg_flows, mode='nearest')
    return w_segs

def single_jacobian_det(flow):
    """
    flow has shape (batch, C, H, W, S)
    """
    flow = flow[None]
    # Compute Jacobian determinant
    batch_size, _, height, width, depth = flow.shape
    dx = flow[:, :, 1:, 1:, 1:] - flow[:, :, :-1, 1:, 1:] + 1
    dy = flow[:, :, 1:, 1:, 1:] - flow[:, :, 1:, :-1, 1:] + 1
    dz = flow[:, :, 1:, 1:, 1:] - flow[:, :, 1:, 1:, :-1] + 1
    jac = np.stack([dx, dy, dz], axis=1).transpose(0, 3, 4, 5, 1, 2)
    det = np.linalg.det(jac)
    return det[0]

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #%%
    fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Sep27_145203_normal_lits_d_.pkl'
    mode = 'normal'
    fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Oct03_210121_masked_lits_d_.pkl'
    mode = 'mask'
    # fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Oct11_183942_soft-m_lits_d_.pkl'
    # mode = 'soft_mask'
    # fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Oct16_164126_normal_o.01_lits_d_.pkl'
    # mode = 'normal_o.01'

    dct = pkl.load(open(fname, 'rb'))
    print(dct.keys())
    print(dct['dice_liver'])
    #%%
    idx = 0
    errs = []

    h5_name = '/home/hynx/regis/recursive-cascaded-networks/datasets/lits_deform_fL.h5'
    reader = FFD_GT(h5_name)

    save_im = True
    for idx in range(len(dct['id1'])):
    # for idx in [0]:
        agg_flow = dct['agg_flows'][idx][-1].permute(1,2,3,0).numpy()
        
        id1 = dct['id1'][idx][7:]
        id2 = dct['id2'][idx]
        print('analyzing', id1, id2)
        reader.set_to(id1, id2);
        gt_flow = reader.gt_flow
        seg2 = reader.seg2
        seg1 = reader.seg1

        w_segs= cal_warped_seg(dct['agg_flows'][idx], torch.tensor(seg2)[None,None].repeat(4,1,1,1,1).float())
        for i, s in enumerate([seg2, *w_segs, seg1]):
            print('{}th layer| organ size {}| tumor size {}'.format(i, (s>0.5).sum(), (s>1.5).sum()))

        masks = [seg2>1.5]
        thres = np.percentile(reader.disp, [90, 95, 99])
        masks = masks + [np.logical_or(seg2>1.5, m) for m in reader.get_masks(thres)]
        err = [eP2P(agg_flow, gt_flow, m) for m in masks]
        print('{} to {} flow end to end error:'.format(id1, id2), err)
        errs.append(err)

        #%% 
        areas = abs(single_jacobian_det(agg_flow.transpose(3,0,1,2)))
        values = np.percentile(areas, [85, 90, 95, 100])
        q_areas = np.digitize(areas, values)
        from tools.utils import plt_img3d_axes
        def func_a(ax, i):
            # p = ax.imshow(np.exp(areas[i]), cmap='jet', alpha=.5)
            # plt.colorbar(p, ax=ax)
            d = ax.contourf(areas[i], levels=values, linewidths=.8)
            # ax.clabel(d, inline=True, fontsize=5)
            ax.imshow(reader.img2[i], cmap='gray', alpha=.5)

        #%% check r_ffd (larger disp than p_ffd)
        r_ffd = reader.r_ffd
        def ffd_flow(ffd, size):
            pt = np.mgrid[0:size[0], 0:size[1], 0:size[2]].astype(np.float32).reshape(3, -1).T
            flow = ffd(pt)-pt
            flow = flow.T.reshape(3,*size)
            return flow
        r_ffd_flow = ffd_flow(r_ffd, seg2.shape)
        p_ffd_flow = ffd_flow(reader.p_ffd, seg2.shape)
        #%%
        # np.histogram(r_ffd_flow)
        show_img(np.linalg.norm(reader.gt_flow, axis=-1))
        # show_img(np.linalg.norm(p_ffd_flow, axis=0))
        # np.histogram(np.linalg.norm(r_ffd_flow, axis=0)),np.histogram(np.linalg.norm(p_ffd_flow, axis=0))

        #%% save image
        if save_im:
            img1 = reader.img1
            img2 = reader.img2
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