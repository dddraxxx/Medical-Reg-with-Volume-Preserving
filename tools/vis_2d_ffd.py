import matplotlib
import numpy as np
from matplotlib import pyplot as plt
# import linecollection
from matplotlib.collections import LineCollection

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
    id1 = '106'
    id2 = '104'
    ffd_reader.set_to(id1, id2)
    #%%
    img1, img2, seg1, seg2 = ffd_reader.img1, ffd_reader.img2, ffd_reader.seg1, ffd_reader.seg2
    gt_flow = ffd_reader.return_flow()
    flow_norm = np.linalg.norm(gt_flow, axis=-1)
    p_disp = ffd_reader.p_disp
    norm_levels = np.percentile(p_disp, [75, 85, 95, 99])
    #%%
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    def plot_single_flow(flow, ax, gt=None, points=None, q_max=None, c_max=None, norm=None):
        fx, fy, fz = flow
        if points is None:
            points = np.mgrid[0:128:128//fx.shape[0], 0:128:128//fx.shape[1]].astype(np.float32)
        y, z = points
        plot_grid(fy+y, fz+z, ax, colors='black', alpha=0.3)
        if gt is not None:
            ax.invert_yaxis()
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
                norm = norm[::norm.shape[0]//fx.shape[0], ::norm.shape[1]//fx.shape[1]]
            cp = ax.contour(y, z, norm, cmap='cool', levels=norm_levels, alpha=1, norm=matplotlib.colors.Normalize(vmin=0, vmax=c_max))
            # show contour number
            ax.clabel(cp, inline=True, fontsize=5)
        if q_max is not None:
            qu = ax.quiver(y, z, fy, fz, fx, angles='xy', scale_units='xy', scale=1, alpha=1, cmap = 'bwr', headwidth=5, norm=matplotlib.colors.Normalize(vmin=-q_max, vmax=q_max))
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
    x,y,z = np.mgrid[0:img1.shape[0], 0:img1.shape[1], 0:img1.shape[2]].astype(np.float32)
    n =8
    # norm_flow = gt_flow / np.linalg.norm(gt_flow, axis=0)
    c_max = abs(gt_flow).max()
    c_max = abs(p_disp).max()
    q_max = np.linalg.norm(gt_flow, axis=-1).max()

    from utils import plt_img3d_axes
    fig = plt.figure(figsize=(30,25))
    # plt.axis('scaled')
    def func_flow(ax, i):
        flow_pts = gt_flow[i]
        flow_pts = flow_pts[::n,::n]
        plot_single_flow(flow_pts.transpose(-1, 0,1), ax, q_max=None, c_max=c_max, norm=p_disp[i])
        ax.imshow(img2[i], cmap='gray', alpha=0.8)
    axes = plt_img3d_axes(img2, func=func_flow, fig=fig)
    fig.subplots_adjust(wspace=0.05, hspace=0.01)
    # fig.tight_layout()
    # put colorbar at desire position
    fig.colorbar(ax=axes, mappable=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=c_max), cmap='cool'), fraction=0.05, pad=0.01)
    fig.colorbar(ax=axes, location='left', mappable=matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-q_max, vmax=q_max), cmap='bwr'), fraction=0.05, pad=0.01)
    # plt.show()
    plt.savefig('flow.png')
# %%
