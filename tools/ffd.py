# %% import functions
import logging
import os
import random
import matplotlib
from matplotlib import widgets
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl
import numpy as np
from pathlib import Path as pa
import pygem as pg
from free_form_deformation import quick_FFD

import sys
sys.path.append('../')
from utils import visualize_3d, tt, show_img, combine_pil_img, plt_img3d_axes, combo_imgs, draw_seg_on_vol
from visualization import plot_landmarks, plt_close

# calculate centroid of seg2
def cal_centroid(seg2):
    obj = seg2==2
    obj = obj.astype(np.float32)
    x, y, z = np.meshgrid(np.arange(obj.shape[0]), np.arange(obj.shape[1]), np.arange(obj.shape[2]), indexing='ij').astype(float)
    c_x = np.sum(x*obj)/np.sum(obj)
    c_y = np.sum(y*obj)/np.sum(obj)
    c_z = np.sum(z*obj)/np.sum(obj)
    print(c_x, c_y, c_z)
    # show_img(img2[:,:, int(c_z)])

# part 1: simulate pressure of the tumor
def find_loc_to_paste_tumor(seg1, kseg2, bbox):
    """
    This function finds a location in `seg1` to paste a tumor region represented by `kseg2` within a bounding box `bbox`. 
    The value '1' in `seg1` represents an organ and the value '2' in `kseg2` represents a tumor. 
    The function returns the bounding box for the location where the tumor can be pasted on `seg1`.
    
    Parameters:
    seg1 (np.ndarray): A 3D array of shape (S, H, W) representing the organ segmentation.
    kseg2 (np.ndarray): A 3D array of shape (S, H, W) representing the tumor segmentation.
    bbox (np.ndarray): A 3D bounding box of shape (2, 3) which defines the size of the region to be pasted on `seg1`.
    
    Returns:
    np.ndarray: A bounding box of shape (2, 3) representing the location where the tumor can be pasted on `seg1`.
    
    Raises:
    ValueError: If it's not possible to find a location to paste the tumor on `seg1`.
    """
    box_length = bbox[1] - bbox[0]
    area =  seg1.shape - box_length
    # how to pick a left up corner?
    box_pts = np.mgrid[0:area[0], 0:area[1], 0:area[2]].reshape(3, -1).astype(int)
    # pick a tumor point in kseg2
    tmr_pt = np.argwhere(kseg2==2)[0]
    organ_pt = (tmr_pt - bbox[0])[:, None] + box_pts
    # organ_pt should be in seg1 organ
    pts = box_pts.T[seg1[tuple(organ_pt.tolist())]>0]
    print('len of pts in box', len(pts))
    pts = pts[::5]
    pts = np.random.permutation(pts)
    if len(pts):
        for i in range(len(pts)):
            pt = pts[i]
            s1 = seg1[pt[0]:pt[0]+box_length[0], pt[1]:pt[1]+box_length[1], pt[2]:pt[2]+box_length[2]]==1
            s2 = kseg2[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], bbox[0][2]:bbox[1][2]]==2
            if (s1&s2 == s2).all():
                return np.array([pt, pt+box_length])
    raise ValueError("cannot find a location to paste tumor")


def find_largest_component(seg, label=2):
    from monai.transforms import KeepLargestConnectedComponent
    k = KeepLargestConnectedComponent(label)
    kseg = np.maximum(k(seg[None].copy()), seg>0)
    return kseg

def ext_bbox(f_bbox, ratio=.5):
    """
    Extend the bounding box by a random amount within the given ratio.

    Parameters:
    f_bbox (numpy array): Initial bounding box with shape (2, 3).
    ratio (float): Ratio of the maximum amount to extend the bounding box.

    Returns:
    numpy array: The extended bounding box with shape (2, 3).

    """
    box_length = f_bbox[1]-f_bbox[0]
    ext_l = np.random.rand(3)*(box_length*ratio)
    print('box length', box_length, '\nextended length', ext_l)
    bbox = np.stack([np.maximum(f_bbox[0]-ext_l, 0), np.minimum(f_bbox[1]+ext_l, 128-1)], axis=0)
    return bbox

def find_bbox(seg1, seg2, total_bbox=True):
    '''
    Param:
        total_bbox: if use the whole image as ffd bbox
    Return:
        kseg2: largest component of seg2
        o_bbox: original bbox of seg2
        f_bbox: bbox of pasting location in seg1
        bbox: bbox for ffd
    '''
    from monai.transforms.utils import generate_spatial_bounding_box
    kseg2 = find_largest_component(seg2)
    bbox = generate_spatial_bounding_box(kseg2, lambda x:x==2)
    kseg2 = kseg2[0]
    o_bbox = np.array(bbox)
    f_bbox = find_loc_to_paste_tumor(seg1, kseg2, o_bbox)
    print("origin bbox", o_bbox)
    if total_bbox:
        bbox = np.array([[0,0,0], seg1.shape])
    else:
        bbox = ext_bbox(f_bbox)
    return kseg2, o_bbox, f_bbox, bbox

def draw_3dbox(res, l, r, val=255):
    res = res.copy()
    res[l[0]:r[0], l[1], l[2]:r[2]] = val
    res[l[0]:r[0], r[1], l[2]:r[2]] = val
    res[l[0]:r[0], l[1]:r[1], l[2]] = val
    res[l[0]:r[0], l[1]:r[1], r[2]] = val
    return res

from scipy.ndimage import map_coordinates
def presure_ffd(bbox, paste_seg2, fct_low=0.5, fct_range=0.4, l=8, order=1):
    global seg_pts, prs_matrix, dst_matrix, dsp_matrix
    ffd = quick_FFD([l, l, l])
    ffd.box_origin = bbox[0]
    ffd.box_length = bbox[1]-bbox[0]
    control_pts = ffd.control_points(False).reshape(l,l,l,3)#[1:-1,1:-1,1:-1]
    # calculate tumor map for each control point
    seg_pts = map_coordinates(paste_seg2, control_pts.reshape(-1,3).T, order=0, mode='nearest')
    seg_pts = seg_pts.reshape(*control_pts.shape[:3])
    # calculate the pressure
    # calculate the distance between control points
    dsp_matrix = control_pts.reshape(-1, 3) - control_pts.reshape(-1, 3)[:, None]
    dsp_matrix /= ffd.box_length
    # diagonal entries are the same point, and no need to calculate the distance
    dst_matrix = (dsp_matrix**2).sum(axis=-1)
    dst_matrix[dst_matrix==0] = 1
    fct = np.random.rand()*fct_range+fct_low
    print('factor', fct)
    # pressure = factor * (1/distance^2),
    # pressure outside organ = 0
    prs_matrix = fct*dsp_matrix/(dst_matrix[...,None]**(1+order/2))
    prs_pts = (prs_matrix*(seg_pts!=2).reshape(1,-1,1)).sum(axis=0)
    dx, dy, dz = prs_pts.reshape(*control_pts.shape).transpose(-1,0,1,2)/((l-1)**3)
    # ffd.array_mu_x[1:-1,1:-1,1:-1] = dx
    # ffd.array_mu_y[1:-1,1:-1,1:-1] = dy
    # ffd.array_mu_z[1:-1,1:-1,1:-1] = dz
    ffd.array_mu_x = dx
    ffd.array_mu_x[seg_pts==0] = 0
    ffd.array_mu_y = dy
    ffd.array_mu_y[seg_pts==0] = 0
    ffd.array_mu_z = dz
    ffd.array_mu_z[seg_pts==0] = 0
    return ffd

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

def ffd_flow(ffd, size, return_mesh=False):
    pt = np.mgrid[0:size[0], 0:size[1], 0:size[2]].astype(np.float32).reshape(3, -1).T
    flow = ffd(pt)-pt
    flow = flow.reshape(*size, 3)
    return flow if not return_mesh else (flow, pt.reshape(*size,3))

def ffd_mesh(ffd, shape=None, mesh=None, return_mesh=False):
    assert shape or mesh
    if shape is not None:
        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
        mesh = np.stack([x, y, z], axis=-1).astype(float)
    if mesh is not None:
        mesh = mesh.astype(float)
    n_mesh = ffd(mesh.reshape(-1, 3))
    nr_mesh = n_mesh.reshape(128,128,128,3)
    return nr_mesh if not return_mesh else (nr_mesh, mesh)

def flow_reverse(flow, rev_points=None):
    if rev_points is None:
        rev_points = np.mgrid[0:flow.shape[0], 0:flow.shape[1], 0:flow.shape[2]].astype(np.float32).reshape(3, -1).T
    xi = rev_points.reshape(-1, 3)
    points = flow.reshape(-1,3) + xi
    values = -flow.reshape(-1,3)
    from scipy.interpolate import griddata
    rev_flow = griddata(points[::], values, xi, method='nearest')
    return rev_flow.reshape(*flow.shape)

def do_ffd(ffd, img, seg=None, order=1, reverse=False):
    """
    Perform Free Form Deformation on the input image.
    
    Parameters:
    - ffd (ndarray): the control points of the deformation grid
    - img (ndarray): the input image
    - seg (ndarray, optional): the segmentation mask of the input image
    - order (int, optional): the order of the interpolation. Default is 1
    - reverse (bool, optional): whether to reverse the deformation. Default is False
    
    Returns:
    - res (ndarray): the deformed image
    - reseg (ndarray, optional): the deformed segmentation mask if seg is provided
    
    """
    if reverse:
        # get flow
        flow, mesh = ffd_flow(ffd, img.shape, True)
        # reverse flow using nearest
        flow_rev = flow_reverse(flow, mesh)
        # get point
        nr_mesh = mesh + flow_rev
    else:
        nr_mesh = ffd_mesh(ffd, img.shape)
    res = map_coordinates(img, nr_mesh.transpose(-1,0,1,2), order=order, mode='nearest')
    # globals().update(locals())
    if seg is not None:
        reseg = map_coordinates(seg, nr_mesh.transpose(-1,0,1,2), order=0, mode='constant', cval=0)
        return res, np.round(reseg).astype(np.uint8)
    return res

def roll_pad(data, shift, padding=0):
    """
    Roll and pad a numpy ndarray along multiple axes with specified padding value.

    Parameters:
    data (ndarray): The input data to be rolled and padded.
    shift (list or tuple): The number of places by which elements are shifted along each axis.
    padding (int or float, optional): The value to fill the newly created padding elements. Default is 0.

    Returns:
    ndarray: The input data after being rolled and padded along multiple axes.
    """
    res = np.roll(data, shift, axis=(0,1,2))
    off_index = lambda x: np.s_[int(x):] if x<0 else np.s_[:int(x)]
    offset = [off_index(i) for i in shift]
    res[offset[0], ...] = padding
    res[:, offset[1]] = padding
    res[..., offset[2]] = padding
    return res

from scipy.ndimage import distance_transform_edt

# signed distance map
def sdm(kseg2, feather_len):
    # global dist_pts, weight_kseg2, dst_to_weight
    dist_pts = distance_transform_edt(kseg2 != 2) - distance_transform_edt(
        kseg2 == 2)
    weight_kseg2 = np.zeros_like(kseg2, dtype=np.float32)
    dst_to_weight = lambda x, l: 0.5 - x / 2 / l
    idx = abs(dist_pts) < feather_len
    weight_kseg2[idx] = dst_to_weight(dist_pts[idx], feather_len)
    weight_kseg2[dist_pts <= -feather_len] = 1
    # feathering area should be inside organ
    inside_organ = kseg2>0
    weight_kseg2[~inside_organ] = 0
    return weight_kseg2

# distance map
def dm(kseg2, feather_len):
    # global dist_pts, weight_kseg2, dst_to_weight
    dist_pts = distance_transform_edt(kseg2!=2)
    weight_kseg2 = np.zeros_like(kseg2, dtype=np.float32)
    dst_to_weight = lambda x, l: 1-x/l
    idx = dist_pts<feather_len
    weight_kseg2[idx] = dst_to_weight(dist_pts[idx], feather_len)
    inside_organ = kseg2>0
    weight_kseg2[~inside_organ] = 0
    return weight_kseg2

def direct_paste(res, img2, kseg2, f_bbox, o_bbox):
    """
    Paste tumor part of img2 to res at f_bbox.

    Parameters:
    res: the image to be pasted
    img2: the image to paste
    kseg2: the segmentation of img2
    f_bbox: the bbox of the tumor in res
    o_bbox: the bbox of the tumor in img2
    
    Returns:
    res1: the image after pasting
    """
    weight_kseg2 = (kseg2>1.5).copy()
    weight_seg1 = roll_pad(1-weight_kseg2, f_bbox[0]-o_bbox[0], padding=1)
    roll_img2 = roll_pad(img2, f_bbox[0]-o_bbox[0], padding=0)
    res1 = res*weight_seg1 + roll_img2*(1-weight_seg1)
    return res1

def feathering_paste(feather_len, res, img2, kseg2, f_bbox, o_bbox):
    weight_kseg2 = sdm(kseg2, feather_len)
    weight_seg1 = roll_pad(1-weight_kseg2, f_bbox[0]-o_bbox[0], padding=1)
    roll_img2 = roll_pad(img2, f_bbox[0]-o_bbox[0], padding=0)
    res1 = res*weight_seg1 + roll_img2*(1-weight_seg1)
    return res1

def calc_disp(points, axis=-1):
    flow = points-np.moveaxis(np.mgrid[:points.shape[0],:points.shape[1],:points.shape[2]].astype(np.float32), 0, axis)
    return np.linalg.norm(flow,axis=axis)

## %% part 2 of the deform: random deformation
def random_ffd(l=5):
    """Generate a random FFD with l control points in each direction"""
    ffd = quick_FFD([l, l, l])
    ffd.box_length = np.full(3, 128-1)
    ffd.box_origin = np.zeros(3)
    # get random field
    x_field, y_field, z_field = (np.random.rand(3, l, l, l)-0.5) /(l-1)
    ffd.array_mu_x += x_field
    ffd.array_mu_y += y_field
    ffd.array_mu_z += z_field
    return ffd

def ffd_params(ffd):
    """Encode ffd parameters to a vector"""
    return np.array([*ffd.n_control_points, *ffd.box_origin, *ffd.box_length, *ffd.array_mu_x.flatten(), *ffd.array_mu_y.flatten(), *ffd.array_mu_z.flatten()])

def get_paste_seg(seg1, kseg2, f_bbox, o_bbox):
    """
    seg1: original seg
    kseg2: seg2 after deform"""
    paste_seg2 = np.zeros_like(kseg2)
    paste_seg2[f_bbox[0][0]:f_bbox[1][0], f_bbox[0][1]:f_bbox[1][1], f_bbox[0][2]:f_bbox[1][2]] = kseg2[o_bbox[0][0]:o_bbox[1][0], o_bbox[0][1]:o_bbox[1][1], o_bbox[0][2]:o_bbox[1][2]]
    paste_seg2 = np.where(paste_seg2==2, 2, seg1)
    return paste_seg2

# TODO: Add foldover check for the flow field
def check_foldover(flow):
    """ Check if the flow field has foldover 
    
    Parameters:
    flow: np.array, HxWx3
    
    Returns:
    foldover: bool, True if has foldover
    """



#%%
if __name__=='__main__':
    # use h5
    import h5py
    fname = '/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits.h5'
    dct = h5py.File(fname, 'r')

    # tumor sorted idx
    no_tumor_idxs = [87,  89, 105, 106,  32,  34,  91, 114, 115,  38,  41,  47, 119]
    sorted_tumor_idx = ([83, 127, 25, 12, 5, 112,  15, 125,  59,  67,  20,  65,  95,  24,  92,  58,  73,  54,   3,  63,  14,  42,  86, 121,  55, 111, 107,  69,  61,  62,  57, 126,  81,  85, 120,  43,  66,  11,  99,  68,  45, 75,  50,  53, 0, 2,  19,  77,  60,  18, 102,  31,  30,  35,  10,  96,   8,   9,  29, 7,   1,  37,  79,   6,  21,  49,  13,  17,  78,  94,  72,  23,  26, 109,  22, 103,  52, 113,  36,  82, 122,  48, 110,  74, 124,  70, 128,  40,  46,  27, 101,  80,  93,  88,  90, 44, 123,  28,  39,  84,  64, 104,  16,  76,  51,  97,  71,  98,  56, 116, 118, 117, 129, 33, 100, 108, 130, 4])
    #%% find tumor size
    tumor_size = []
    for i in sorted_tumor_idx:
        tumor_size.append(np.sum(dct[f'{i}']['segmentation'][...]>1.5))
    # print(list(enumerate(tumor_size)))
    #%% select ids
    # no tumor instance
    id1s = ['115']
    # with tumor (equally sample 10)
    id2s = [str(i) for i in sorted_tumor_idx[50::3]]
    #%% below tumor too large
    # id2s = ['129', "33", "100", "108", "130", "4"]
    write_h5 = True
    if write_h5:
        # h5_writer = h5py.File('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits_deform.h5', 'w')
        h5_writer = h5py.File('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits_paste.h5', 'w')
        print('writing to h5', h5_writer.filename)
    img_dir = '/home/hynx/regis/FFD_imgs/paste'
    pa(img_dir).mkdir(exist_ok=True)
    (pa(img_dir)/'img').mkdir(exist_ok=True)
    (pa(img_dir)/'deform').mkdir(exist_ok=True)
    #%%
    for id1 in id1s[:]:
        for id2 in id2s[:5]:
            print('\ndeforming organ {} for tumor {}'.format(id1, id2))
            img1 = dct[id1]['volume'][...]
            img2 = dct[id2]['volume'][...]
            seg1 = dct[id1]['segmentation'][...]
            seg2 = dct[id2]['segmentation'][...]
            print('{} has tumor size: {}'.format(id2, np.sum(seg2==2)))

            combine_pil_img(show_img(seg1), show_img(seg2), show_img(find_largest_component(seg2)[0])).save(pa(img_dir)/'img/{}-{}_seg.png'.format(id1, id2))
            combine_pil_img(show_img(img1), show_img(img2)).save(pa(img_dir)/'img/{}-{}_img.png'.format(id1, id2))
            #@%% stage 1
            try:
                kseg2, o_bbox, f_bbox, bbox = find_bbox(seg1, seg2, True)
            except ValueError as e:
                print(e)
                continue
            paste_seg2 = get_paste_seg(seg1, kseg2, f_bbox, o_bbox)
            #%% do presure ffd
            def presure_ffd(bbox, paste_seg2, fct_low=0.5, fct_range=0.4, l=8, order=1, sample_num=1000):
                ffd = quick_FFD([l, l, l])
                ffd.box_origin = bbox[0]
                ffd.box_length = bbox[1]-bbox[0]
                control_pts = ffd.control_points(False).reshape(l,l,l,3)#[1:-1,1:-1,1:-1]
                # calculate tumor map for each control point
                seg_pts = map_coordinates(paste_seg2, control_pts.reshape(-1,3).T, order=0, mode='nearest')
                seg_pts = seg_pts.reshape(*control_pts.shape[:3])
                # calculate the pressure
                # sample tumor points
                tumor_pts = np.mgrid[:paste_seg2.shape[0], :paste_seg2.shape[1], :paste_seg2.shape[2]].transpose(1,2,3,0)[paste_seg2==2]
                tumor_pts = tumor_pts[np.random.choice(tumor_pts.shape[0], sample_num, replace=False)] if tumor_pts.shape[0]>sample_num else tumor_pts
                # calculate the distance between control points
                dsp_matrix = control_pts.reshape(-1, 3) - tumor_pts.reshape(-1, 3)[:, None]
                dsp_matrix /= ffd.box_length
                # diagonal entries are the same point, and no need to calculate the distance
                dst_matrix = (dsp_matrix**2).sum(axis=-1)
                dst_matrix[dst_matrix==0] = 1
                prs_matrix = dsp_matrix/(1e-5+dst_matrix[...,None]**(0.5+order/2))
                prs_pts = (prs_matrix*(seg_pts!=2).reshape(1,-1,1)).sum(axis=0)/sample_num
                dxyz = prs_pts.reshape(*control_pts.shape).transpose(-1,0,1,2)
                # normalize the pressure by points' distance to boundary
                dist_to_bnd, bnd_ind = distance_transform_edt(seg_pts>0, sampling=1/(l-1), return_indices=True)
                dist_to_tu = distance_transform_edt(seg_pts<2, sampling=1/(l-1))
                disp_prs = np.linalg.norm(prs_pts, axis=-1)
                size_fct = (dist_to_bnd.flatten())/(disp_prs+1e-9) #*fct
                factor = np.percentile(size_fct[seg_pts.flatten()>0],
                    40)
                dxyz = dxyz * factor
                # smooth/regularize the pressure field
                b2t_ratio = dist_to_bnd/(dist_to_tu+dist_to_bnd+1e-6)
                dxyz = dxyz * b2t_ratio[None]
                print('max pressure', np.linalg.norm(dxyz, axis=0).max())
                dx, dy, dz = dxyz
                ffd.array_mu_x = dx
                ffd.array_mu_y = dy
                ffd.array_mu_z = dz
                # pressure outside organ = 0
                ffd.array_mu_x[seg_pts==0] = 0
                ffd.array_mu_y[seg_pts==0] = 0
                ffd.array_mu_z[seg_pts==0] = 0
                globals().update(locals())
                return ffd
            if True:
                p_ffd = presure_ffd(bbox, paste_seg2, fct_low=0.5, fct_range=0, l=16, order=3.5, sample_num=1000)
                wpts = ffd_mesh(p_ffd, img1.shape)
                disp = calc_disp(wpts, axis=-1)
                print('max displacement', disp.max())
                # visualize pressure to the image
                def vis_pressure():
                    plt.figure()
                    ax = plt.gca()
                    def func(ax, i):
                        d = ax.contour(disp[i], levels=3)
                        ax.clabel(d, inline=True, fontsize=3)
                        ax.imshow(paste_seg2[i], alpha=0.6)
                    plt_img3d_axes(img1, func)
                    plt.savefig(img_dir+'img/{}-{}_disp.png'.format(id1, id2))
                    plt.show()
                res, res_seg = do_ffd(p_ffd, img1, seg1, reverse=True)
            else:
                res = img1.copy()
                res_seg = paste_seg2.copy()
            #%% paste tumor
            res1 = direct_paste(res, img2, kseg2, f_bbox, o_bbox)
            # feather_len = 5
            # res1 = feathering_paste(feather_len, res, img2, kseg2, f_bbox, o_bbox)
            res_seg[paste_seg2==2]=2
            # save img in stage1
            combo_imgs(res1, res_seg, draw_seg_on_vol(res1, res_seg)
                       ).save(pa(img_dir)/'deform/{}-{}_stage1.png'.format(id1, id2))
            logging.info('stage1 done and picture saved to {}'.format(pa(img_dir)/'deform/{}-{}_stage1.png'.format(id1, id2)))
            #%% stage 2
            def stage2(res1, res_seg, p_ffd=lambda x:x, r_ffd=None,):
                if r_ffd is None:
                    r_ffd = random_ffd(l=3) # l=5 the default settings
                # plt_ffd(ffd)
                # check if the ffd needs to be reversed
                res2, res2_seg = do_ffd(r_ffd, res1, res_seg, reverse=True)
                mesh_pts = np.mgrid[0:128, 0:128, 0:128].reshape(3, -1).T.astype(float)
                # below the composition of flow
                ffd_pts = p_ffd(r_ffd(mesh_pts))
                ffd_pts = ffd_pts.reshape(128, 128, 128, 3)
                # res_gt = map_coordinates(res2, ffd_pts.transpose(-1,0,1,2), order=1, mode='nearest', cval=0)
                return res2, res2_seg, ffd_pts, r_ffd
            res2, res2_seg, ffd_pts, r_ffd = stage2(res1, res_seg)
            nt_res2, nt_res2_seg, nt_ffd_pts, nt_r_ffd = stage2(img1, seg1, r_ffd=r_ffd)
            
            # res2, res2_seg = res1, res_seg
            res2, nt_res2 = np.clip(res2,0,255), np.clip(nt_res2,0,255)
            combo_imgs((res2), res2_seg, draw_seg_on_vol((res2), res2_seg)
                       ).save(pa(img_dir)/'deform/{}-{}_stage2.png'.format(id1, id2))
            combo_imgs((nt_res2), nt_res2_seg, draw_seg_on_vol((nt_res2), nt_res2_seg)
                       ).save(pa(img_dir)/'deform/{}-{}_stage2_nt.png'.format(id1, id2)) 
            logging.info('stage2 done and picture saved to {}'.format(pa(img_dir)/'deform/{}-{}_stage2.png'.format(id1, id2)))

            #%% save img1, res2, ffd_gt, and segmentation of res2 to h5 file
            # create group if not exist
            def write_h5_(h5_writer, id1, img1, seg1, id2=None, res2=None, res2_seg=None, ffd_pts=None, r_ffd=None, deformed=True, points=None):
                if not deformed:
                    if id1 not in h5_writer:
                        h5_writer.create_group(id1)
                        h5_writer[id1].create_dataset('volume', data=(img1), dtype='uint8')
                        h5_writer[id1].create_dataset('segmentation', data=seg1, dtype='uint8')
                        if points is not None:
                            h5_writer[id1].create_dataset('point', data=points, dtype='uint8')
                else:
                    assert all([id1, img1, seg1, id2, res2, res2_seg, ffd_pts, r_ffd])
                    if id1 not in h5_writer:
                        h5_writer.create_group(id1)
                        h5_writer[id1].create_dataset('volume', data=(img1), dtype='uint8')
                        h5_writer[id1].create_dataset('segmentation', data=seg1, dtype='uint8')
                    if 'id2' not in h5_writer[id1]:
                        h5_writer[id1].create_group('id2')
                    h5_writer[id1]['id2'].create_group(id2)
                    h5_writer[id1]['id2'][id2].create_dataset('volume', data=(res2), dtype='uint8')
                    h5_writer[id1]['id2'][id2].create_dataset('segmentation', data=res2_seg, dtype='uint8')
                    h5_writer[id1]['id2'][id2].create_dataset('ffd_gt', data=ffd_pts.astype(np.float32))
                    # save ffd in h5
                    ffd_obj = np.concatenate([ffd_params(p_ffd), ffd_params(r_ffd)])
                    h5_writer[id1]['id2'][id2].create_dataset('ffd', data=ffd_obj)

            #%% also preserve the landmark points if any
            deformed_points = None
            import json
            lmk = json.load(open('/home/hynx/regis/recursive-cascaded-networks/landmark_json/lits17_landmark.json', 'r'))
            if id1 in lmk:
                points = np.array(lmk[id1])
                deformed_points = np.full(points.shape, -1)
                for i in range(points.shape[0]):
                    if points[i,0] == -1:
                        continue
                    deformed_points[i] = ffd_pts[points[i,0], points[i,1], points[i,2]]
                plot_landmarks(res1, deformed_points, save_path=pa(img_dir)/'deform/{}-{}_points.png'.format(id1, id2))
                plt_close()
                
            if write_h5:
                # write_h5_(h5_writer, id1, img1, seg1, id2, res2, res2_seg, ffd_pts, r_ffd, deformed=True)
                write_h5_(h5_writer, '{}'.format(id1), img1, seg1, deformed=False, points=points)
                write_h5_(h5_writer, 'p_{}_{}'.format(id1, id2), res2, res2_seg, deformed=False, points=deformed_points)
                write_h5_(h5_writer, 'nt_{}_{}'.format(id1, id2), nt_res2, nt_res2_seg, deformed=False, points=deformed_points)

                
        # plot_landmarks(img1, points, save_path=pa(img_dir)/'deform/{}_points.png'.format(id1), color='r')
        # plt_close()

    if write_h5: h5_writer.close()