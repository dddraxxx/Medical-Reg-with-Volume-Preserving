# Review two change happened when we transform dicom to h5

#%%
import json
import nibabel as nib
import numpy as np
from utils import getLargestCC, bbox_seg

# let's see step by step if the landmarks is right
def get_lmk_file(pid):
    """ read file by id P1-1/..."""
    # pid like P1-1
    root_dir = "/home/hynx/regis/recursive_cascaded_networks/data_process/brain_lmks"
    json_file = root_dir + f"/{pid[:2]}/" + pid + ".mrk.json"
    return json_file

def read_lmks(jsp=""):
    # markups -> controlPoints -> [{id, position}]
    jsn = json.load(open(jsp))
    points = []
    for p in jsn['markups'][0]['controlPoints']:
        points.append({
            "id": p['id'],
            "position": p['position']
        })
    return points

def get_tumorseg_striped_nii(pid):
    return '/home/hynx/regis/helpful_repo/zdongzhong/nifti/brain/seg_unet/{}.nii.gz'.format(pid)
def get_striped_nii(pid):
    return '/home/hynx/regis/helpful_repo/zdongzhong/nifti/brain/striped/{}.nii.gz'.format(pid)
def get_original_nii(pid):
    return '/home/hynx/regis/helpful_repo/zdongzhong/nifti/brain/original/{}.nii.gz'.format(pid)
def get_striped_nii(pid):
    return '/home/hynx/regis/helpful_repo/zdongzhong/nifti/brain/striped/{}.nii.gz'.format(pid)
def get_resize_factor(orig_size, reshape_size):
    return reshape_size / orig_size
def get_crop_box(seg, pad):
    """ crop the seg to the bbox

    Returns:
        bbox: [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    l_seg = getLargestCC(seg)
    bbox = bbox_seg(l_seg)
    bbox = bbox + np.array([-pad, -pad, -pad, pad, pad, pad])
    min_xyz = np.maximum(bbox[:3], 0)
    max_xyz = np.minimum(bbox[3:], seg.shape)
    return np.concatenate([min_xyz, max_xyz])
def convert_to_cropcoord(coord, bbox):
    min_xyz = bbox[:3]
    crop_coord = coord - min_xyz
    return crop_coord

import matplotlib.pyplot as plt
def draw_img_point(img, pt):
    fig = plt.figure()
    ax = fig.gca()

    # draw the point
    pt = np.round(pt).astype(int)
    ax.scatter(pt[2], pt[1], c='r', s=10)

    # draw the image
    img = img[pt[0], :, :]
    ax.imshow(img, cmap='gray')
    return ax

class LMK:
    """ Use LPS coordinate system"""
    def __init__(self, pid):
        self.pid = pid
        self.lmk_file = get_lmk_file(pid)
        self.lps_lmks = read_lmks(self.lmk_file)

    def convert_to_pixel(self, nii):
        """ convert lps_lmks to pixel coordinate in nii
        Args:
            nii: nibabel object
        """
        self.pixel_coord = {}
        # find the pixel coordinate of point in nii_0
        for lmk in self.lps_lmks:
            lps_point = np.array(lmk['position'])
            coord_RAS = [-lps_point[0], -lps_point[1], lps_point[2]]
            pixel_coord = np.round(np.linalg.inv(nii.affine).dot(coord_RAS + [1]))[:3]
            self.pixel_coord.update({lmk['id']: pixel_coord})
        return self.pixel_coord

    def convert_to_cropcoord(self, bbox):
        """ convert pixel_coord to crop_coord
        Args:
            bbox: [min_x, min_y, min_z, max_x, max_y, max_z]
        """
        self.crop_coord = {}
        for lmk in self.pixel_coord:
            pixel_coord = self.pixel_coord[lmk]
            min_xyz = bbox[:3]
            crop_coord = pixel_coord - min_xyz
            self.crop_coord.update({lmk: crop_coord})
        return self.crop_coord

    def convert_to_resizecoord(self, bbox, reshape_size):
        """ convert crop_coord to resize_coord
        Args:
            reshape_size: [x, y, z]
        """
        self.resize_coord = {}
        for lmk in self.crop_coord:
            crop_coord = self.crop_coord[lmk]
            fac = get_resize_factor(bbox[3:] - bbox[:3] + 1, reshape_size)
            self.resize_coord.update({lmk: crop_coord * fac})
        return self.resize_coord

    def convert(self):
        """ convert lps_lmks to pixel_coord, crop_coord, resize_coord"""
        # 1. convert to pixel_coord
        nii = nib.load(get_original_nii(self.pid))
        self.convert_to_pixel(nii)
        # 2. convert to crop_coord
        striped = nib.load(get_striped_nii(self.pid)).get_fdata()
        seg_strip = striped > striped.min()
        bbox = get_crop_box(seg_strip, 20)
        self.convert_to_cropcoord(bbox)
        # 3. convert to resize_coord
        reshape_size = np.array([128, 128, 128])
        self.convert_to_resizecoord(bbox, reshape_size)
        return self.resize_coord


lmks_target = ["P1-1", "P1-2"]
lmks = [LMK(pid) for pid in lmks_target]
# print(lmks[0].lps_lmks)

import h5py
def get_h5_arr(pid):
    h5f = '/home/hynx/regis/recursive_cascaded_networks/datasets/brain_test_23.h5'
    with h5py.File(h5f, 'r') as f:
        return f[pid[1:]]['volume'][...]

h5arr = get_h5_arr(lmks_target[0])
print(h5arr.shape)

lmks[0].convert()
#%%
res_coord = lmks[0].resize_coord['2']
draw_img_point(h5arr, res_coord)
#%%
# 1. We verify if the orginial nii has the correct landmark position


import nibabel as nib
import numpy as np
nii_0 = nib.load(get_original_nii(lmks_target[0]))
# print(nii_0)
lps_point = np.array(lmks[0].lps_lmks[2]['position'])
# find the pixel coordinate of point in nii_0
coord_RAS = [-lps_point[0], -lps_point[1], lps_point[2]]
pixel_coord = np.round(np.linalg.inv(nii_0.affine).dot(coord_RAS + [1]))[:3]

# fit (261,319,15) in 3d_slicer
print("pixel coord: ", pixel_coord) # output [15, 319, 261]


pt = pixel_coord
nii_0arr = nii_0.get_fdata()

# from monai.visualize import matshow3d
# matshow3d(nii_0.get_fdata(), every_n=5)

#%% 2. we verify that we can get the correct pixel coordinate from h5

# we using seg_mask to find the transformation


# transformation: get largerCC --> bbox --> crop (padding) --> resize
# reload utils
import importlib
from importlib import reload
import utils
reload(utils)
from utils import getLargestCC, bbox_seg

img_arr = nib.load(get_striped_nii(lmks_target[0])).get_fdata()
seg_arr = img_arr > img_arr.min()
bbox0 = get_crop_box(seg_arr, 20)
print('box size: ', bbox0[3:] - bbox0[:3])
crop_coord = convert_to_cropcoord(pixel_coord, bbox0)
print("crop coord: ", crop_coord)

# verify the crop coord
ax = draw_img_point(nii_0arr, pt)
bbox = get_crop_box(seg_arr, 0)
ax.plot(bbox[2], bbox[1], c='r', marker='o', markersize=10)
ax.plot(bbox[5], bbox[4], c='r', marker='o', markersize=10)
ax.imshow(seg_arr[int(pt[0]), :, :], cmap='gray', alpha=0.5)

# get resize factor
box_size = np.array(bbox0[3:] - bbox0[:3] + 1)
reshape_size = np.array([128, 128, 128])

resize_factor = get_resize_factor(box_size, reshape_size)
res_coord = crop_coord * resize_factor
print("resized coord: ", res_coord)