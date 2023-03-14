from monai.metrics.utils import get_surface_distance
#%%
# from nnunet.preprocessing.preprocessing import resample_data_or_seg
# from adet.utils.visualize_niigz import *
from skimage.transform import resize
import h5py
# import SimpleITK as sitk
import numpy as np
import pickle as pkl
import pydicom
import monai
from monai.transforms import (
    Compose,
    LoadImage,
    SpatialResample,
    Spacing,
    ResampleToMatch,
    Orientation,
)
from pathlib import Path as pa

from tqdm import tqdm
from utils import combo_imgs, draw_seg_on_vol

# read npz file
def get_npz(path):
    return np.load(path)['data']

# read nii file
def read_nii(path, with_meta=None, meta=False, original_meta=False):
    ori = Orientation(as_closest_canonical=True)
    img = LoadImage(simple_keys=True, image_only=True)(path)
    img.meta = with_meta if with_meta else img.meta
    o_ma = img.meta
    # print(o_ma['affine'])
    img = ori(img[None])[0]
    if meta and original_meta:
        ma = img.meta
        return np.array(img), ma, o_ma
    if meta:
        ma = img.meta
        return np.array(img), ma
    if original_meta:
        return np.array(img), o_ma
    return np.array(img)

def read_dicom_itk(path):
    names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(path))
    # Read the DICOM series
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(names)
    img = reader.Execute()
    return img

def read_dicom(path):
    plots = []
    slices = [pydicom.dcmread(f) for f in pa(path).glob("*.dcm")]
    for ds in sorted(slices, key = lambda x: int(x.InstanceNumber), reverse = True):
        pix = ds.pixel_array
        pix = pix*1+(-1024)
        plots.append(pix)

    y = np.dstack(plots)
    return y
#%%
def normalize_ct(arr, property_f = '/mnt/sdb/nnUNet/nnUNet_cropped_data/Task029_LITS/dataset_properties.pkl'):
    # borrow from nnunet preprocessed stats; read pkl
    with open(property_f, 'rb') as f:
        it_prop = pkl.load(f)['intensityproperties'][0]#['local_props']['train_{}'.format(id)]
    # clip by it_prop bounds
    arr = np.clip(arr, it_prop['percentile_00_5'], it_prop['percentile_99_5'])
    # z-norm by it_prop mean and std
    arr = (arr - it_prop['mean']) / it_prop['sd']
    return arr

def normalize_mri(data, seg, non_zero=True):
    if non_zero:
        mask = seg >= 0
        data[mask] = (data[mask] - data[mask].mean()) / (data[mask].std() + 1e-8)
        data[mask == 0] = data[mask].min()-0.1
    else:
        mn = data.mean()
        std = data.std()
        # print(data.shape, data.dtype, mn, std)
        data = (data - mn) / (std + 1e-8)
    return data

# get id and find seg file
def get_seg(id, root_dir = '/mnt/sdc/lits/train', **kwargs):
    seg_id = f'segmentation-{id}.nii'
    seg_path = root_dir + '/' + seg_id
    return read_nii(seg_path, **kwargs)

# get id and find volume file
def get_data(id, root_dir = '/mnt/sdc/lits/train', **kwargs):
    ct_id = f'volume-{id}.nii'
    ct_path = root_dir + '/' + ct_id
    return read_nii(ct_path, **kwargs)

# crop seg by the bounding box where value is larger than 0
def bbox_seg(seg):
    bbox = seg.nonzero()
    bbox = np.array([np.min(bbox[0]), np.min(bbox[1]), np.min(bbox[2]), np.max(bbox[0]), np.max(bbox[1]), np.max(bbox[2])])
    return bbox
def crop(seg, bbox, pad=5):
    bbox = bbox.copy()
    # increase bbox by pad
    bbox[0] = max(bbox[0] - pad, 0)
    bbox[1] = max(bbox[1] - pad, 0)
    bbox[2] = max(bbox[2] - pad, 0)
    bbox[3] += pad
    bbox[4] += pad
    bbox[5] += pad
    return seg[bbox[0]:bbox[3]+1, bbox[1]:bbox[4]+1, bbox[2]:bbox[5]+1]

def find_no_tumor_h5(h5_path):
    with h5py.File(h5_path, 'r') as f:
        tumour_sum = []
        keys = list(f.keys())
        for id in sorted(keys, key=lambda x: int(x)):
            tumour_sum.append(np.sum(np.array(f[id]['segmentation'])==2))
            if np.max(f[id]['segmentation'])<2:
                print(id)
    print(np.argsort(tumour_sum))
    print(tumour_sum)

def visualize_3d(img, save_name, label=None, **kwargs):
    from monai.visualize.utils import matshow3d, blend_images
    from matplotlib import pyplot as plt
    # matplotlib use same size as original size
    if kwargs.get('original_size', False):
        h, w = img.shape[-2:]
        import matplotlib as mpl
        dpi = mpl.rcParams['figure.dpi']
        print(w, h, img.shape)
        figsize = (6*w/dpi, (img.shape[-3]//5+1)*h/dpi)
        kwargs['figsize'] = figsize
        kwargs.pop('original_size')
    if label is not None:
        if len(label.shape) == 3:
            label = label[None]
        if len(img.shape) == 3:
            img = img[None]
        img = blend_images(img, label, alpha=0.3)
        channel_dim = 0
        matshow3d(img, channel_dim=channel_dim, every_n=5, frames_per_row=5, cmap='gray', margin=1, frame_dim=-3, **kwargs)
    else: matshow3d(img, every_n=5, frames_per_row=5, cmap='gray', margin=1, frame_dim=-3, **kwargs)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.close()

def visual_h5(h5_path):
    dir_path = pa('images') / pa(h5_path).stem
    # get full path
    dir_path = dir_path.resolve()
    # mkdir
    dir_path.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, 'r') as f:
        bar = tqdm(f)
        for id in bar:
            img = np.array(f[id]['volume']).astype(np.float32)
            seg = np.array(f[id]['segmentation']).astype(np.float32)
            # visualize_3d(img, save_name = dir / f'{id}_img.jpg', figsize=(10, 10))
            # visualize_3d(seg, save_name = dir / f'{id}_seg.jpg', figsize=(10, 10))
            # # overlay img and seg
            # visualize_3d(img[None ], save_name = dir / f'{id}_img_seg.jpg', label=seg[None], figsize=(10, 10))
            import torch
            img = torch.from_numpy(img)
            seg = torch.from_numpy(seg)
            combo_imgs(img, seg, draw_seg_on_vol(img, seg.long(), to_onehot=True)
                       ).save(dir_path / f'{id}_img.jpg')
            bar.set_description(f'visualizing {id} as' + f' {dir_path / f"{id}_img.jpg"}')

# visualize original ct data
def visual_nii(id, path = '/mnt/sdc/lits/train'):
    seg = get_seg(id, path)
    data = get_data(id, path)
    data = normalize_ct(data)
    # visulize_3d(data, save_name = f'images/{id}_img.jpg')
    # visulize_3d(seg, save_name = f'images/{id}_seg.jpg')
    # calculate centroid of mass
    obj = seg==2
    import scipy.ndimage as ndi
    centroid = ndi.measurements.center_of_mass(obj)
    centroid = np.array(centroid).astype(int)
    print(centroid)
    print(data.shape)

    radius = 96
    # crop
    c_data = data[int(centroid[0])-radius:int(centroid[0])+radius, int(centroid[1])-radius:int(centroid[1])+radius, int(centroid[2])-radius:int(centroid[2])+radius]
    # normalize
    c_data = (c_data - np.min(c_data)) / (np.max(c_data) - np.min(c_data))*255
    c_data = c_data.astype(np.uint8)
    c_seg = seg[int(centroid[0])-radius:int(centroid[0])+radius, int(centroid[1])-radius:int(centroid[1])+radius, int(centroid[2])-radius:int(centroid[2])+radius]
    # visulize_3d(c_data, save_name = f'images/{id}_c_img.jpg')
    from PIL import Image
    img_x = Image.fromarray(c_data[radius, :, :]).convert('L')
    img_y = Image.fromarray(c_data[:, radius, :]).convert('L')
    img_z = Image.fromarray(c_data[:, :, radius]).convert('L')
    img_x.save(f'images/{id}_c_img_x.jpg')
    img_y.save(f'images/{id}_c_img_y.jpg')
    img_z.save(f'images/{id}_c_img_z.jpg')

    cl_data = draw_seg_on_vol(c_data[None], (c_seg==2)[None]).permute(0,2,3,1).numpy()
    cl_data = (cl_data*255).astype(np.uint8)
    print(cl_data.shape)
    cl_img_x = Image.fromarray(cl_data[radius, :, :])
    cl_img_y = Image.fromarray(cl_data[:, radius, :])
    cl_img_z = Image.fromarray(cl_data[:, :, radius])
    cl_img_x.save(f'images/{id}_cl_img_x.jpg')
    cl_img_y.save(f'images/{id}_cl_img_y.jpg')
    cl_img_z.save(f'images/{id}_cl_img_z.jpg')

    print('make grid')
    # put imgs in grid together
    def make_grid_pil(*ims, nrow=3, padding=2):
        ncol = (nrow-1+len(ims)) // nrow
        ims = [im.convert('RGB') for im in ims]
        widths, heights = zip(*(i.size for i in ims))
        total_width = ncol * max(widths) + (ncol-1) * padding
        max_height = nrow * max(heights) + (nrow-1) * padding
        new_im = Image.new('RGB', (total_width, max_height), color='white')
        for i, im in enumerate(ims):
            col = i // nrow
            row = i % nrow
            x_offset = col * (max(widths) + padding)
            y_offset = row * (max(heights) + padding)
            new_im.paste(im, (x_offset, y_offset))
        return new_im
    im = make_grid_pil(cl_img_x, cl_img_y, cl_img_z, img_x, img_y, img_z, padding=5)
    im.save(f'images/{id}_grid.jpg')

def resize_data(data, shape, is_seg=False):
    """
    Resize data to shape (H, W, D)
    
    Args:
        data: numpy array of shape (H, W, D)
        shape: tuple of shape (H, W, D)
        is_seg: bool, if True, use order=0, else use order=3

    Returns:
        data: numpy array of shape (H, W, D)
    """
    if is_seg:
        data = resize(data, shape, order=0, preserve_range=True, anti_aliasing=False)
    else:
        data = resize(data, shape, order=3, preserve_range=True, anti_aliasing=False, mode='edge')
    return data

# assign seg according to group id/segmentation in hdf5 file
def store_segs_lits(h5_path, selected=np.s_[:]):

    ids = range(131)
    # create h5 and put data and seg in it
    with h5py.File(h5_path, 'w') as f:
        for id in ids[selected]:
            print('id', id)
            data, meta, o_meta = get_data(id, with_meta=None, original_meta=True, meta=True)
            seg = get_seg(id, with_meta=o_meta)
            data = normalize_ct(data)
            
            bbox = bbox_seg(seg)
            seg = crop(seg, bbox, pad=5)
            data = crop(data, bbox, pad=5)

            seg = resize_data(seg, (128, 128, 128), is_seg=True)
            data = resize_data(data, (128, 128, 128), is_seg=False)

            # reverse axis according to affine
            assert meta['space'] == 'RAS'
            # affine = meta['affine']
            # reverse_ax = affine.diagonal()[:3] < 0
            # reverse_idx = tuple(reverse_ax.nonzero().flatten().tolist())
            # print(affine.diagonal()[:3], reverse_idx)
            # data = np.flip(data, axis = reverse_idx)
            # seg = np.flip(seg, axis = reverse_idx)
            if (seg[:64]>0).sum() > (seg[64:]>0).sum(): # in cases that liver on the left
                print('reverse left-right in no.{}'.format(id))
                data = np.flip(data, axis = 0)
                seg = np.flip(seg, axis = 0)
            # visulize_3d(data, inter_dst=3, save_name = 'img.jpg')
            group = f'{id}'
            # create group
            f.create_group(group)
            # normalize volume to 0-255
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
            data = data.astype(np.uint8)
            seg = seg.astype(np.uint8)
            # store data and seg as 'volume' and 'segmentation' dataset
            f[group].create_dataset('volume', data=data, dtype='uint8')
            f[group].create_dataset('segmentation', data=seg, dtype='uint8')

# assign seg according to group id/segmentation in hdf5 file
def store_segs_brats(h5_path, selected=np.s_[:], cropped_data='/mnt/sdb/nnUNet/nnUNet_cropped_data/Task082_BraTS2020/', mod=1, lab=1):
    """
    We select T1 modality for training. Every data is resized to (128, 128, 128).
    """
    all_data = sorted(pa(cropped_data).glob('Bra*.npz'))
    ids = range(1, len(all_data)+1)
    import os
    h5_path = os.path.realpath(h5_path)
    # create h5 and put data and seg in it
    with h5py.File(h5_path, 'w') as f:
        for id in ids[selected]:
        # for id in ids[selected][:5]:
            print('id', id)
            data_path = all_data[id-1]
            data = np.load(data_path)['data']
            seg = data[-1]
            img = data[mod]
            seg = resize_data(seg, (128, 128, 128), is_seg=True)
            img = resize_data(img, (128, 128, 128), is_seg=False)

            # remove background, and set tumor to 2
            img = normalize_mri(img, seg)
            if isinstance(lab, int):
                seg[seg==lab] = -10
            else:
                for l in lab:
                    seg[seg==l] = -10
            seg[seg>=0] = 1
            seg[seg==-10] = 2
            seg[seg==-1] = 0

            data = img
            # visulize_3d(data, inter_dst=3, save_name = 'img.jpg')
            group = f'{id}'
            # create group
            f.create_group(group)
            # normalize volume to 0-255
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
            data = data.astype(np.uint8)
            seg = seg.astype(np.uint8)
            # store data and seg as 'volume' and 'segmentation' dataset
            f[group].create_dataset('volume', data=data, dtype='uint8')
            f[group].create_dataset('segmentation', data=seg, dtype='uint8')

def store_segs_mrbr(h5_path, selected=np.s_[:], cropped_data='/home/hynx/regis/zMrbrains18/training', mod='reg_T1.nii.gz'):
    """
    We select T1 modality for training. Every data is resized to (128, 128, 128).
    """
    all_data = sorted(pa(cropped_data).glob('**/pre/'+mod))
    print(all_data)
    ids = range(1, len(all_data)+1)
    import os
    h5_path = os.path.realpath(h5_path)
    def read_data(path):
        import nibabel as nib
        seg_path = pa(path).parent.parent/'segm.nii.gz'
        seg = nib.load(seg_path).get_fdata()
        arr_path = path
        arr = nib.load(arr_path).get_fdata()
        arr = arr.transpose(2, 1, 0)
        seg = seg.transpose(2, 1, 0)
        return arr, seg
    # create h5 and put data and seg in it
    with h5py.File(h5_path, 'w') as f:
        for id in ids[selected]:
        # for id in ids[selected][:5]:
            print('id', id)
            data_path = all_data[id-1]
            img, seg = read_data(data_path)
            seg = resize_data(seg, (128, 128, 128), is_seg=True)
            img = resize_data(img, (128, 128, 128), is_seg=False)

            seg[seg!=0]=seg[seg!=0]+2

            data = img
            # visulize_3d(data, inter_dst=3, save_name = 'img.jpg')
            group = f'{id}'
            # create group
            f.create_group(group)
            # normalize volume to 0-255
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
            data = data.astype(np.uint8)
            seg = seg.astype(np.uint8)
            # store data and seg as 'volume' and 'segmentation' dataset
            f[group].create_dataset('volume', data=data, dtype='uint8')
            f[group].create_dataset('segmentation', data=seg, dtype='uint8')

def get_meta(id, path='/mnt/sdc/lits/train'):
    img, meta = LoadImage()(f'{path}/volume-{id}.nii')
    return meta

# you only need to run the function once
def modify_h5_orientation(h5_path, path='/mnt/sdc/lits/train'):
    with h5py.File(h5_path, 'r+') as f:
        for id in f.keys():
            print('id', id)
            meta = get_meta(id, path)
            affine = meta['affine']
            assert meta['space'] == 'RAS'
            print(meta['space'])
            # reverse_ax = affine.diagonal()[:3] < 0
            # volume = np.array(f[id]['volume'])
            # seg = np.array(f[id]['segmentation'])
            # reverse_idx = tuple(reverse_ax.nonzero().flatten().tolist())
            # print(affine.diagonal(), reverse_idx)
            # volume = np.flip(volume, axis = reverse_idx)
            # seg = np.flip(seg, axis = reverse_idx)
            # f[id]['volume'][:] = volume
            # f[id]['segmentation'][:] = seg

def visual_dicom(path = '/mnt/sdc/qhd/nbia/Duke-Breast-Cancer-MRI'):
    img_dir = 'images/breast'
    pa(img_dir).mkdir(parents=True, exist_ok=True)
    for id in range(1,922):
        dir = f'{path}/Breast_MRI_{id:03d}/'
        print(dir)
        try:
            dir = pa(dir).glob('*/*ax t1*').__next__()
        except Exception as e:
            print(e)
            continue
        img = read_dicom(dir).transpose(0,1,2)
        print(np.unique(img), img.shape)
        # img = normalize_ct(img)
        img = resize_data(img, (128, 128, 128), is_seg=False)
        visulize_3d(img, save_name = f'{img_dir}/{id}_img.jpg')
        print(img.shape)
        break

def store_dicom_h5(h5_path, path = '/mnt/sdc/qhd/nbia/Duke-Breast-Cancer-MRI'):
    def get_data_by_id(id):
        dir = f'{path}/Breast_MRI_{id:03d}/'
        try:
            dir = pa(dir).glob('*/*ax t1*').__next__()
        except Exception as e:
            print(e)
            return
        img = read_dicom(dir)
        return img
    ds = range(1,922)
    with h5py.File(h5_path, 'w') as f:
        for id in ids:
            print('id', id)
            data = get_data_by_id(id)
            if data is None:
                continue
            data = resize_data(data, (128, 128, 128), is_seg=False)
            data = data.transpose(2, 1, 0)[::-1].copy()
            if id%100==1:
                visulize_3d(data, inter_dst=2, save_name = 'img.jpg')

            group = f'{id}'
            # create group
            f.create_group(group)
            # normalize volume to 0-255
            data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
            data = data.astype(np.uint8)
            # store data and seg as 'volume' and 'segmentation' dataset
            f[group].create_dataset('volume', data=data, dtype='uint8')
            f[group].create_dataset('segmentation', data=np.zeros((128,128,128)), dtype='uint8')

if __name__=='__main__':
    # store_segs('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits.h5')#, selected=np.s_[128:129])
    # store_segs_mrbr('/home/hynx/regis/recursive-cascaded-networks/datasets/mrbr.h5')
    # store_segs_mrbr('/home/hynx/regis/recursive-cascaded-networks/datasets/brats_t1ce.h5')
    # visual_h5('/home/hynx/regis/recursive-cascaded-networks/datasets/brain_test_23.h5')
    visual_h5('/home/hynx/regis/recursive-cascaded-networks/datasets/liver_test10.h5')
    # visual_h5('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits.h5')
    # visual_h5('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lspig_val.h5')
    # visual_h5('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits_paste.h5')

    # data, meta = read_nii(path='/mnt/sdc/lits/train/volume-51.nii', original_meta=True)
    # seg = read_nii(path='/mnt/sdc/lits/train/segmentation-51.nii', with_meta=meta)
    # # print(meta)
    # data = normalize_ct(data)
    # bbox = bbox_seg(seg)
    # seg = crop(seg, bbox, pad=5)
    # data = crop(data, bbox, pad=5)
    # print(data.shape, seg.shape, np.unique(seg))
    # visualize_3d(data, label=seg, save_name='img.jpg', figsize=(20,60))
    # find_no_tumor_h5('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits.h5')
    # visual_dicom()
    # store_dicom_h5('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/breast.h5')
    import torch
    def visualize_nii(id, path='/mnt/sdc/lits/train'):
        # img, meta = LoadImage()(f'{path}/volume-{id}.nii')
        import nibabel as nib
        from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform
        imgn : nib.Nifti1Image = nib.load(f'{path}/volume-{id}.nii')
        seg, meta = LoadImage()(f'{path}/segmentation-{id}.nii')
        print('be4', nib.aff2axcodes(imgn.affine))
        orig_ornt = io_orientation(imgn.affine)
        targ_ornt = axcodes2ornt("RAS")
        transform = ornt_transform(orig_ornt, targ_ornt)
        imgn = imgn.as_reoriented(transform)

        img = np.array(imgn.dataobj)
        seg = seg.numpy()
        img = normalize_ct(img)
        bbox = bbox_seg(seg)
        print(bbox)
        print('after', nib.aff2axcodes(imgn.affine))
        from pprint import pprint
        meta.update({'id':id})
        pprint(meta, stream=open('meta.txt', 'a'))
        # img = img[bbox[0]:bbox[3], bbox[1]:bbox[4], bbox[2]:bbox[5]]
        img = img[:,:,bbox[2]:bbox[5]]
        img = torch.from_numpy(img)
        # aff = meta['affine']
        # flips = aff.diagonal()[:3]
        # img = img.flip(0) if flips[0] < 0 else img
        # img = img.flip(1) if flips[1] < 0 else img
        # img = img.flip(2) if flips[2] < 0 else img
        save_dir = 'images/lits_orig'
        # create save_dir
        pa(save_dir).mkdir(parents=True, exist_ok=True)
        visualize_3d(img.permute(2,0,1), save_name = f'{save_dir}/{id}_img.jpg')
    # for id in [39,128]:
    #     visualize_nii(id)
    # modify_h5_orientation('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits.h5')