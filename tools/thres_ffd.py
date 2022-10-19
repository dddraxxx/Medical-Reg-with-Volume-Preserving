#%%
from typing import List
import h5py
import pygem as pg
from free_form_deformation import quick_FFD
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata


def ffd_from_params(params):
    ffds = []
    while len(params):
        l0, l1, l2 = params[:3].astype(int)
        # ffd = pg.FFD([l0, l1, l2])
        ffd = quick_FFD([l0, l1, l2])
        ffd.box_origin = params[3:6]
        ffd.box_length = params[6:9]
        ffd.array_mu_x = params[9:9+l0*l1*l2].reshape(l0, l1, l2)
        ffd.array_mu_y = params[9+l0*l1*l2:9+2*l0*l1*l2].reshape(l0, l1, l2)
        ffd.array_mu_z = params[9+2*l0*l1*l2:9+3*l0*l1*l2].reshape(l0, l1, l2)
        ffds.append(ffd)
        params = params[9+3*l0*l1*l2:]
    return ffds

class FFD_GT:
    def __init__(self, h5py_file):
        self.reader  = h5py.File(h5py_file, 'r')

    def set_to(self, id1, id2):
        self.id1 = id1
        self.id2 = id2
        self.ffd_gt = self.reader[id1]['id2'][id2]['ffd_gt'][...]
        self.shape = self.ffd_gt.shape
        ffd_params = self.reader[id1]['id2'][id2]['ffd'][...]
        self.p_ffd, self.r_ffd = ffd_from_params(ffd_params)

        # get displacement in the first stage
        pt = np.mgrid[0:self.shape[0], 0:self.shape[1], 0:self.shape[2]].astype(np.float32).reshape(3, -1).T
        rpt = self.r_ffd(pt)
        disp = self.p_ffd(rpt)-rpt
        disp = disp.reshape(self.shape)
        self.disp = np.linalg.norm(disp, axis=-1)

        # get warping gt
        points = self.ffd_gt.reshape(-1,3)
        values = -self.return_flow().reshape(-1,3)
        xi = np.mgrid[0:img2.shape[0], 0:img2.shape[1], 0:img2.shape[2]].astype(np.float32).reshape(3, -1).T
        self.gt_flow = griddata(points, values, xi, method='nearest')
        # get threshold
        self.thres = np.percentile(self.disp, 95)
        return self
    
    def validate(self):
        '''validate p_ffd and r_ffd correspond to ffd_gt'''
        pt = np.mgrid[0:self.shape[0], 0:self.shape[1], 0:self.shape[2]].astype(np.float32).reshape(3, -1).T
        ffd_re = self.p_ffd(self.r_ffd(pt))
        ffd_gt = self.ffd_gt.reshape(ffd_re.shape)
        assert np.allclose(ffd_re, ffd_gt)
    
    def return_flow(self):
        '''return flow of ffd_gt'''
        return self.ffd_gt - np.moveaxis(np.mgrid[0:self.shape[0], 0:self.shape[1], 0:self.shape[2]], 0, -1)
    
    def __getattr__(self, name: str):
        if name in self.__dict__:
            return self.__dict__[name]
        id1 = self.id1
        id2 = self.id2
        seg2 = self.reader[id1]['id2'][id2]['segmentation']
        img2 = self.reader[id1]['id2'][id2]['volume']
        seg1 = self.reader[id1]['segmentation']
        img1 = self.reader[id1]['volume']
        if name == 'seg2':
            return seg2[...]
        elif name == 'img2':
            return img2[...]
        elif name == 'seg1':
            return seg1[...]
        elif name == 'img1':
            return img1[...]
    
    def get_masks(self, thres: List):
        return [self.disp>t for t in thres]
    
    def close(self):
        self.reader.close()

#%% main function
if __name__ == '__main__':
    from utils import show_img, combine_pil_img, combo_imgs, plt_img3d_show
    #%%
    # get ffd params
    ffd_reader = FFD_GT('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits_deform_fL.h5')
    id1 = '119'
    id2 = '51'
    ffd_reader.set_to(id1, id2)
    #%% validate if loaded right
    validate = False
    if validate:
        ffd_reader.validate()

    #%%
    draw = False
    if draw:
        # %matplotlib widget
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from utils import combine_pil_img, plt_grid3d

        ffd = ffd_reader.p_ffd
        fig = plt.figure(figsize=(4, 4))
        ax = Axes3D(fig)
        # ax.scatter(*control_pts[seg_pts].T, s=20, c='green')
        # ax.scatter(*control_pts[~seg_pts].T, s=20)
        ax.scatter(*ffd.control_points().T, s=5, c='red')
        # plt_grid3d(ffd.control_points(False), ax, colors='b', linewidths=1)
        plt_grid3d(ffd.control_points(), ax, colors='r', linewidths=0.5)
        # show x,y,z axis
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(3, 90)
        plt.show()
    #%% validate gt warping
    import torch
    import sys
    sys.path.append('..')
    sys.path.append('.')
    from networks.spatial_transformer import SpatialTransform
    def cal_single_warped(flow, img):
        img = torch.tensor(img).double()
        st = SpatialTransform(img.shape)
        w_img = st(img[None,None], flow[None], mode='bilinear')
        return w_img[0,0]
    
    def do_warp(flow, img, seg=None, order=1):
        x, y, z = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2]), indexing='ij')
        mesh = np.stack([x, y, z], axis=-1)
        nr_mesh = flow + mesh
        # this is backward pass, not correct
        res = map_coordinates(img, nr_mesh.transpose(-1,0,1,2), order=order, mode='constant', cval=0)
        if seg is not None:
            reseg = map_coordinates(seg, nr_mesh.transpose(-1,0,1,2), order=0, mode='constant', cval=0)
            return res, reseg
        return res
    
    def ffd_flow(ffd, size):
        pt = np.mgrid[0:size[0], 0:size[1], 0:size[2]].astype(np.float32).reshape(3, -1).T
        flow = ffd(pt)-pt
        flow = flow.T.reshape(3,*size)
        return flow

    def do_ffd(ffd, img, seg=None):
        # flow = ffd_flow(ffd, img.shape).transpose(1,2,3,0).astype(float)
        # return do_warp(flow, img, seg)
        x, y, z = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2]), indexing='ij')
        mesh = np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(float)
        n_mesh = ffd(mesh)
        nr_mesh = n_mesh.reshape(128,128,128,3)
        # this is backward pass, not correct
        res = map_coordinates(img, nr_mesh.transpose(-1,0,1,2), order=3, mode='constant', cval=0)
        if seg is not None:
            reseg = map_coordinates(seg, nr_mesh.transpose(-1,0,1,2), order=0, mode='constant', cval=0)
            return res, reseg
        return res
    #%% validate if gt_flow is img1 -> img2
    gt_flow = ffd_reader.return_flow()
    img2 = ffd_reader.img2
    img1 = ffd_reader.img1
    def validate_flow():
        gt_flow_torch = torch.tensor(gt_flow).permute(3,0,1,2)
        w_img2 = cal_single_warped(gt_flow_torch, img1)
    # %% calculate reverse of gt_flow (img2 -> img1)
    points = ffd_reader.ffd_gt.reshape(-1,3)
    values = -gt_flow.reshape(-1,3)
    xi = np.mgrid[0:img2.shape[0], 0:img2.shape[1], 0:img2.shape[2]].astype(np.float32).reshape(3, -1).T
    from scipy.interpolate import griddata
    # Lienar too long, use nearest
    # %timeit rev_flow = griddata(points[::], values, xi, method='lienar')
    rev_flow = griddata(points[::], values, xi, method='nearest')
    #%% visualize rev_flow similarity
    def visualize_rev_flow():
        rev_flow_t = torch.tensor(rev_flow.reshape(128,128,128,3)).permute(3,0,1,2)
        plt_img3d_show((img1-cal_single_warped(rev_flow_t, img2).numpy()), cmap='jet')
    #%% get pressured area
    # wp_img1 = do_ffd(ffd_reader.p_ffd, ffd_reader.img1)
    p_disp = ffd_reader.disp
    thres = np.percentile(ffd_reader.disp, [90, 95, 99])
    print('thres:', thres)
    def func(ax, i):
        # p = ax.imshow(p_disp[i], cmap='jet', alpha=0.6)
        # plt.colorbar(p, ax=ax)
        ax.imshow(img2[i], cmap='gray', alpha=1)
        d = ax.contour(p_disp[i], levels=thres, linewidths=.8)
        ax.clabel(d, inline=True, fontsize=5)
    import matplotlib.pyplot as plt
    def plt_img3d_axes(imgs, func, intrv=5):
        fig, axes = plt.subplots((len(imgs+1))//(intrv**2), intrv)
        for ax, i in zip(axes.flatten(), range(0, len(imgs), intrv)):
            func(ax, i)
            # turnoff axis
            ax.axis('off')
            # title
            ax.set_title(f'{i}')
        # tight
        plt.tight_layout()
        plt.show()
    plt_img3d_axes(img2, func)

    #%% visualize it and visualize the flow
    from utils import show_img, combine_pil_img, draw_seg_on_vol
    from PIL import Image

    masks = [(ffd_reader.disp>t) for t in thres]
    im = combine_pil_img(show_img(ffd_reader.disp), show_img(ffd_reader.img2), *[show_img(draw_seg_on_vol(ffd_reader.img2[None], (ffd_reader.disp>i)[None], colors = ['pink'])) for i in thres])
    im.save('disp.png')
    # show_img(img1)

    #%%
    ffd_reader.close()