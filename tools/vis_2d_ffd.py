import matplotlib
import numpy as np
from matplotlib import pyplot as plt
# import linecollection
from matplotlib.collections import LineCollection
from utils import cal_rev_flow, cal_single_warped

def plot_single_flow(flow, ax, gt=None, points=None, q_max=None, c_max=None, norm=None, norm_levels=None):
    fx, fy, fz = flow
    if points is None:
        points = np.mgrid[0:128:128//fx.shape[0], 0:128:128//fx.shape[1]].astype(np.float32)
    y, z = points
    plot_grid(y=fy+y, x=fz+z, ax=ax, colors='black', alpha=0.3)
    if gt is not None:
        # ax.invert_yaxis()
        px, py, pz = fx[gt==1], fy[gt==1], fz[gt==1]
        ax.scatter(pz, py, alpha=0.5, c='g')
        px, py, pz = fx[gt==2], fy[gt==2], fz[gt==2]
        ax.scatter(pz, py, alpha=0.5, c='r')
        # cmap1 = matplotlib.colors.ListedColormap(['none', 'green', 'red'])
        # eps = 1e-2
        # ax.contourf(gt.transpose(-1,-2), [0.5, 1.5], cmap=cmap1, alpha=0.8)
    if c_max is not None:
        if norm is None:
            norm = np.linalg.norm(flow, axis=0) 
        else:
            if points is not None:
                cy, cz = np.mgrid[:norm.shape[0], :norm.shape[1]].astype(np.float32)
            else:
                cy, cz = y, z
                norm = norm[::norm.shape[0]//fx.shape[0], ::norm.shape[1]//fx.shape[1]]
        cp = ax.contour(cz, cy, norm, cmap='cool', levels=norm_levels, alpha=1, norm=matplotlib.colors.Normalize(vmin=0, vmax=c_max))
        # show contour number
        ax.clabel(cp, inline=True)
    if q_max is not None:
        qu = ax.quiver(z,y,fz,fy,fx, angles='xy', scale_units='xy', scale=1, alpha=1, cmap = 'bwr', headwidth=5, norm=matplotlib.colors.Normalize(vmin=-q_max, vmax=q_max))
    # ax.axis('off')
    # show colorbar
    # return cp, qu

def plot_grid(x,y, ax=None, **kwargs):
    ax = ax or plt.gca()
    segs1 = np.stack((x,y), axis=2)
    segs2 = segs1.transpose(1,0,2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()
    return ax
#%%
if __name__ == '__main__':
    import torch
    import sys
    sys.path.append('..')
    sys.path.append('.')
    from utils import show_img, combine_pil_img, combo_imgs, plt_img3d_show, plt_img3d_axes
    from vis_3d_ffd import FFD_GT
    #%%
    # get ffd params
    ffd_reader = FFD_GT('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits_deform_fL.h5')
    img_dir = '/home/hynx/regis/FFD_imgs'
    # mkdir
    from pathlib import Path as pa
    pa(img_dir).mkdir(parents=True, exist_ok=True)
    pa(img_dir+'/flow_field').mkdir(parents=True, exist_ok=True)
    mode = 'gt'

    fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Sep27_145203_normal_lits_d_.pkl'
    mode = 'normal'
    # fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Oct03_210121_masked_lits_d_.pkl'
    # mode = 'mask'
    # fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Oct21_164855_mask_karea_lits_d_.pkl'
    # mode = 'mask_karea'
    # below: bad performance
    # fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Oct11_183942_soft-m_lits_d_.pkl'
    # mode = 'soft_mask'
    # fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Oct16_164126_normal_o.01_lits_d_.pkl'
    # mode = 'normal_o.01'
    import pickle as pkl
    dct = pkl.load(open(fname, 'rb'))
    print(dct.keys())
    print(dct['dice_liver'])

    for id1 in ffd_reader.reader:
        for id2 in ffd_reader.reader[id1]['id2']:
            print('Doing {},{}'.format(id1, id2))
            #%%
            ffd_reader.set_to(id1, id2)
            #%%
            img1, img2, seg1, seg2 = ffd_reader.img1, ffd_reader.img2, ffd_reader.seg1, ffd_reader.seg2
            gt_flow = ffd_reader.return_flow()
            # gt_flow[...,0]=0
            # rev_flow = ffd_reader.set_rev_flow()
            if mode=='gt':
                rev_flow = cal_rev_flow(gt_flow)
                # rev_flow[..., 0]=0
                flow_norm = np.linalg.norm(rev_flow, axis=-1)
                p_disp = ffd_reader.p_disp
            else:
                idx=None
                for cnt in range(len(dct['id1'])):
                    r = dct['id1'][cnt].rfind('_')
                    if id1==dct['id1'][cnt][r+1:] and id2==dct['id2'][cnt]:
                        idx=cnt
                agg_flow = dct['agg_flows'][idx][-1].permute(1,2,3,0).numpy()
                rev_flow = cal_rev_flow(agg_flow)
                p_disp = np.linalg.norm(agg_flow, axis=-1)
            norm_levels = np.percentile(p_disp, [90, 95, 99])
            print('Norm levels', norm_levels)

            #%%
            import numpy as np
            import matplotlib.pyplot as plt
            import matplotlib
            
            x,y,z = np.mgrid[0:img1.shape[0], 0:img1.shape[1], 0:img1.shape[2]].astype(np.float32)
            n =8
            c_max = abs(rev_flow).max()
            c_max = abs(p_disp).max()
            q_max = np.linalg.norm(rev_flow, axis=-1).max()
            q_max = rev_flow[...,0].max()

            from utils import plt_img3d_axes
            fig = plt.figure(figsize=(24,20))
            # plt.axis('scaled')
            def func_flow(ax, i):
                flow_pts = rev_flow[i]
                flow_pts = flow_pts[::n,::n]
                plot_single_flow(flow_pts.transpose(-1, 0,1), ax, q_max=q_max, c_max=c_max, norm=p_disp[i], norm_levels=norm_levels)
                ax.imshow(img2[i], cmap='gray', alpha=0.8)
            axes, _ = plt_img3d_axes(img2, func=func_flow, fig=fig)
            fig.subplots_adjust(wspace=0.05, hspace=0.05)
            # fig.tight_layout()
            # put colorbar at desire position
            fig.colorbar(ax=axes, mappable=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=c_max), cmap='cool'), fraction=0.05, pad=0.01)
            fig.colorbar(ax=axes, location='left', mappable=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-q_max, vmax=q_max), cmap='bwr'), fraction=0.05, pad=0.01)
            # plt.show()
            if mode=='gt':
                plt.savefig(img_dir+'/flow_field/{}_{}.jpg'.format(id1,id2))
                #%%
                # cal warped img
                fig = plt.figure(figsize=(24,20))
                w_img2 = cal_single_warped(gt_flow.transpose(-1,0,1,2), img2)
                def func_show(ax, i):
                    ax.imshow(w_img2[i], cmap='gray')
                fig.subplots_adjust(wspace=0.05, hspace=0.05)
                plt_img3d_axes(img2, func=func_show, fig=fig)
                # plt save
                plt.savefig(img_dir+'/flow_field/{}_{}_w.jpg'.format(id1,id2))
            else:
                plt.savefig(img_dir+'/flow_field/{}_{}_{}.jpg'.format(id1,id2, mode))
# %%