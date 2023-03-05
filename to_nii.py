#%%
import pydicom
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path as pa
from tools.utils import *
import numpy as np

def read_dicom(path):
    plots = []
    # slices = [pydicom.dcmread(f) for f in pa(path).glob("*.dcm")]
    slices = [pydicom.dcmread(f) for f in pa(path).glob("IM*")]
    for ds in sorted(slices, key = lambda x: int(x.InstanceNumber), reverse = True):
        pix = ds.pixel_array
        pix = pix*1+(-1024)
        plots.append(pix)

    y = np.dstack(plots)
    return y
# convert to nii
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    Spacingd,
    EnsureTyped,
)

val_transforms = Compose(
    [
        LoadImaged(keys=["image"], ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
        # Orientationd(keys=["image"], axcodes="SAL"),
        Spacingd(
            keys=["image"],
            pixdim=(2., 1.5, 1.5),
            mode=("bilinear"),
        ),
        EnsureTyped(keys=["image"], device='cpu', track_meta=True),
    ]
)
# dicom to nii for better processing
if True:
    root_liver = '/home/hynx/regis/zdongzhong/lm'
    for i1, i, in enumerate(sorted(pa(root_liver).glob('P*'))):
        for i2,j in enumerate(sorted(pa(i).glob('ST*'))):
            print()
            reader = sitk.ImageSeriesReader()
            dc_name = str(j) + '/SE9'
            while pa(dc_name).exists() == False:
                dc_name = dc_name[:-1] +\
                    str(int(dc_name[-1]) - 1)

            # print(i1, i2)
            i1 = str(i)[str(i).find('P')+1:]
            i2 = str(j)[str(j).find('ST')+2:]
            # if i1!='5' or i2!='1': continue
            i1 = int(i1)
            i2 = int(i2)
            print(i1, i2)
            print('reading', dc_name)
            
            if i1==5:
                nii_name = '/home/hynx/regis/zdongzhong/nifti/liver/P{}-{}.nii.gz'.format(i1, i2)

                dicom_names = reader.GetGDCMSeriesFileNames(dc_name)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()

                # # Added a call to PermuteAxes to change the axes of the data
                image = sitk.PermuteAxes(image, [2, 1, 0])
                print('data shape', image.GetSize())

                sitk.WriteImage(image, nii_name)

            if i1==5 and i2==1:
                # read nii and change it
                img = nib.load(nii_name)
                nii_name = '/home/hynx/regis/zdongzhong/nifti/liver_ts/P{}-{}_0001.nii.gz'.format(i1, i2)
                affine = img.affine
                img = img.get_fdata()
                img = img[:-50]
                print(img.shape)
                img = nib.Nifti1Image(img, affine)
                img.to_filename(nii_name)
            elif i1 ==5 and i2==0:
                img = nib.load(nii_name)
                nii_name = '/home/hynx/regis/zdongzhong/nifti/liver_ts/P{}-{}_0001.nii.gz'.format(i1, i2)
                affine = img.affine
                img = img.get_fdata()
                img = img[50:]
                print(img.shape)
                img = np.flip(img, axis=0)
                img = nib.Nifti1Image(img, affine)
                img.to_filename(nii_name)
            else: continue
            input = val_transforms({'image': nii_name})
            # read image affine after transform
            print(input['image_meta_dict']['affine'])
            # get image orientation
            import pprint
            pprint.pprint(input['image_meta_dict'])
            img_dir = '/home/hynx/regis/zdongzhong/images/liver_nii'
            pa(img_dir).mkdir(exist_ok=True)
            

            show_img(input['image'][0], inter_dst=10).save(f'{img_dir}/P{i1}-{i2}.png')
            print('saved image to', f'{img_dir}/P{i1}-{i2}.png')
            # # exit()
            # print('data shape', input['image'].shape)
            # save to npy
            # np.save('/home/hynx/regis/dongzhong/nifti/liver/P{}-{}.npy'.format(i1, i2), input['image'][0])

# %% read_nii
import nibabel as nib
# resize to 64x64x64
import torch.nn.functional as F
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    Orientationd,
    EnsureTyped,
)

val_transforms = Compose(
    [
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        EnsureTyped(keys=["image","label"], device='cpu', track_meta=True),
    ]
)
from tools.utils import *
from pathlib import Path as pa
import numpy as np
from skimage.measure import label   

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC
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

def orient(data):
    return data.permute(2,1,0).flip((0,1))
def write_h5(f, id, img, seg):
    '''
    ensure size is right
    '''
    img = np.array(img).astype(np.float32)
    seg = np.array(seg)
    data = img
    group = f'{id}'
    # create group
    if group not in f:
        f.create_group(group)
    # normalize volume to 0-255
    data = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    data = data.astype(np.uint8)
    seg = seg.astype(np.uint8)
    # store data and seg as 'volume' and 'segmentation' dataset
    f[group].create_dataset('volume', data=data, dtype='uint8')
    f[group].create_dataset('segmentation', data=seg, dtype='uint8')
# for 1-4: just 

import h5py
h5file = '/home/hynx/regis/zdongzhong/nifti/liver_test.h5'
if not pa(h5file).exists():
    h5file = h5py.File(h5file, 'w')
for t1, t1_seg in zip(
    sorted(pa('/home/hynx/regis/zdongzhong/nifti/liver_ts').glob('*.nii.gz')),
    sorted(pa('/home/hynx/regis/zdongzhong/nifti/liver_seg').glob('*.nii.gz'))
):
    print(t1, t1_seg)
    data_name = pa(t1_seg)
    t1 = nib.load(t1).get_fdata()
    t1_seg = nib.load(t1_seg).get_fdata()
    t1 = torch.from_numpy(t1)
    t1_seg = torch.from_numpy(t1_seg)
    # choose largest connected component
    l_seg = getLargestCC(t1_seg>0)
    bbox = bbox_seg(l_seg)
    t1_seg[~l_seg] = 0
    t1_seg = crop(t1_seg, bbox, 5)
    t1 = crop(t1, bbox, 5)
    print(t1.shape, t1_seg.shape)
    # if 'P5' not in str(data_name):
    t1 = orient(t1)
    t1_seg = orient(t1_seg)
    t1 = F.interpolate(t1.unsqueeze(0).unsqueeze(0), size=(128,128,128), mode='trilinear').squeeze(0).squeeze(0)
    t1_seg = F.interpolate(t1_seg.unsqueeze(0).unsqueeze(0), size=(128,128,128), mode='nearest').squeeze(0).squeeze(0)
    print(t1.shape, t1_seg.shape)

    dct = val_transforms({"image": t1, "label": t1_seg})
    dt1, dt1_seg = dct['image'], dct['label']
    print(dt1.shape, dt1_seg.shape)
    combo_imgs(dt1, dt1_seg, axis=0, idst=5).save('/home/hynx/regis/zdongzhong/images/liver/{}_combo.png'.format(data_name.name[:-7]))
    seg_on_img = draw_seg_on_vol(dt1, dt1_seg.long(), inter_dst=5, to_onehot=True)
    show_img(seg_on_img, norm=False, inter_dst=1).save('/home/hynx/regis/zdongzhong/images/liver/{}.png'.format(data_name.name[:-7]))

    # save h5
    # write_h5(h5file, data_name.name[1:4], dt1, dt1_seg)

#%% rename cases
if False:
    from pathlib import Path as pa
    for i in pa('/home/hynx/regis/dongzhong/nifti/liver').glob('*.nii.gz'):
        # copy to parent
        print(i)
        new_name = i.name[:-7]+'_0000.nii.gz'
        # copy and rename
        import shutil
        shutil.copy2(i, '/home/hynx/regis/dongzhong/nifti/liver_nnuent/'+new_name)


