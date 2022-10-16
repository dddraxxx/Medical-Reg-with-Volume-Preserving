#%%
from typing import List
import h5py
import pygem as pg
import numpy as np
from scipy.ndimage import map_coordinates


def ffd_from_params(params):
    ffds = []
    while len(params):
        l0, l1, l2 = params[:3].astype(int)
        ffd = pg.FFD([l0, l1, l2])
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
        return self
    
    def validate(self):
        pt = np.mgrid[0:self.shape[0], 0:self.shape[1], 0:self.shape[2]].astype(np.float32).reshape(3, -1).T
        ffd_re = self.p_ffd(self.r_ffd(pt))
        ffd_gt = self.ffd_gt.reshape(ffd_re.shape)
        assert np.allclose(ffd_re, ffd_gt)
    
    def return_flow(self):
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
    #%%
    # get ffd params
    ffd_reader = FFD_GT('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits_deform_L.h5')
    id1 = '106'
    id2 = '104'
    ffd_reader.set_to(id1, id2)
    #%% validate if loaded right
    validate = False
    if validate:
        ffd_reader.validate()

    #%% get the mask
    np.histogram(ffd_reader.disp, bins=100)
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
    from utils import show_img, combine_pil_img
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
    
    def do_warp(flow, img, seg=None):
        x, y, z = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]), np.arange(img.shape[2]), indexing='ij')
        mesh = np.stack([x, y, z], axis=-1)
        nr_mesh = flow + mesh
        # this is backward pass, not correct
        res = map_coordinates(img, nr_mesh.transpose(-1,0,1,2), order=1, mode='constant', cval=0)
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

    gt_flow = ffd_reader.return_flow()
    gt_flow_torch = torch.tensor(gt_flow).permute(3,0,1,2)
    img2 = ffd_reader.img2
    img1 = ffd_reader.img1
    w_img2 = cal_single_warped(gt_flow_torch, img2)
    # combine_pil_img(show_img(ffd_reader.img1), show_img(ffd_reader.img2), show_img(w_img2.numpy()))
    wp_img2 = do_ffd(ffd_reader.p_ffd, ffd_reader.img1)
    wr_img2 = do_ffd(ffd_reader.r_ffd, wp_img2)
    wr2_img2 = cal_single_warped(gt_flow_torch, img1)
    # wr1_img2 = do_warp(gt_flow, img1)
    combine_pil_img(show_img(ffd_reader.img1), show_img(ffd_reader.img2), show_img(wr_img2), show_img(wr2_img2))
    #%% store it
    thres = np.percentile(ffd_reader.disp, [95, 97, 99])
    print('thres:', thres)
    masks = [(ffd_reader.disp>t) for t in thres]
    #%% visualize it and visualize the flow
    from utils import show_img, combine_pil_img
    from PIL import Image
    from adet.utils.visualize_niigz import *

    im = combine_pil_img(show_img(ffd_reader.disp), show_img(ffd_reader.img2), *[show_img(draw_seg_on_vol(ffd_reader.img2[None], (ffd_reader.disp>i)[None], colors = ['pink'])) for i in thres])
    im.save('disp.png')
    # show_img(img1)

    #%%
    ffd_reader.close()