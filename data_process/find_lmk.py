# Review two change happened when we transform dicom to h5

#%%
import json

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from utils import bbox_seg, draw_img_point, getLargestCC


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

class LMK:
    """ Store the landmarks of a patient

    Properties:
        lps_lmks: use LPS coordinate system in nii
        pixel_coord: use pixel coordinate system in nii
        crop_coord: coordinate after cropping
        resize_coord: coordinate after resizing
        h5_coord: coordinate in h5 file
    """
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

            # we need to consider align_corners in torch.nn.functional.interpolate
            ali_crop_coord = crop_coord + 0.5
            orig_size = bbox[3:] - bbox[:3] + 1
            after_size = reshape_size
            fac = after_size / orig_size
            ali_resize_coord = ali_crop_coord * fac - 0.5

            self.resize_coord.update({lmk: ali_resize_coord})
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
        self.h5_coord = self.resize_coord
        return self.resize_coord


#%%
import h5py
def get_h5_arr(pid):
    h5f = '/home/hynx/regis/recursive_cascaded_networks/datasets/brain_test_23.h5'
    with h5py.File(h5f, 'r') as f:
        return f[pid[1:]]['volume'][...]

lmks_target = np.array([["P1-1", "P1-2"],
               ["P2-1", "P2-2"],])
lmks = []
for l in lmks_target:
    i1, i2 = l
    lmks.append([LMK(i1), LMK(i2)])
    lmks[-1][0].convert()
    lmks[-1][1].convert()

#%%
if __name__ == '__main__':
    def visualize_P(i):
        lmksi = lmks[i]
        lmks_targeti = lmks_target[i]
        for i in range(len(lmks)):
            h5arr = get_h5_arr(lmks_targeti[i])
            res_coord = lmksi[i].resize_coord['1']
            draw_img_point(h5arr, res_coord)
    visualize_P(1)
