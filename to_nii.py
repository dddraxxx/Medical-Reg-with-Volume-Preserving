#%% get nii.gz
import ants
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
    NormalizeIntensityd,
)

li_val_transforms = Compose(
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
br_val_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureTyped(keys=["image"]),
        # Orientationd(keys=["image"], axcodes="RAS"),
        CropForegroundd(keys=["image"], source_key="image"),
        # Spacingd(
        #     keys=["image"],
        #     pixdim=(3., 1.0, 1.0),
        #     mode=("bilinear"),
        # ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)
# dicom to nii for better processing
if True:
    def li_niigze(i,j):
        reader = sitk.ImageSeriesReader()
        dc_name=str(j)
        # li
        # dc_name = str(j) + '/SE9'
        # while pa(dc_name).exists() == False:
        #     dc_name = dc_name[:-1] +\
        #         str(int(dc_name[-1]) - 1)
        # i1 = str(i)[str(i).find('P')+1]
        # i2 = str(j)[str(j).find('ST')+2:]

        # if i1==5 and i2==1:
            #     # read nii and change it
            #     img = nib.load(nii_name)
            #     nii_name = '/home/hynx/regis/zdongzhong/nifti/liver_ts/P{}-{}_0001.nii.gz'.format(i1, i2)
            #     affine = img.affine
            #     img = img.get_fdata()
            #     img = img[:-50]
            #     print(img.shape)
            #     img = nib.Nifti1Image(img, affine)
            #     img.to_filename(nii_name)
            # elif i1 ==5 and i2==0:
            #     img = nib.load(nii_name)
            #     nii_name = '/home/hynx/regis/zdongzhong/nifti/liver_ts/P{}-{}_0001.nii.gz'.format(i1, i2)
            #     affine = img.affine
            #     img = img.get_fdata()
            #     img = img[50:]
            #     print(img.shape)
            #     img = np.flip(img, axis=0)
            #     img = nib.Nifti1Image(img, affine)
            #     img.to_filename(nii_name)
            # else: continue

        input = br_val_transforms({'image': nii_name})
        # read image affine after transform
        # print(input['image_meta_dict']['affine'])
        print(input['image'].shape)
        print(input['image_meta_dict']['affine'])
        # spacing
        print(input['image_meta_dict']['pixdim'])
        # get image orientation
        import pprint
        # pprint.pprint(input['image_meta_dict'])
        img_dir = '/home/hynx/regis/zdongzhong/images/brain_nii'
        pa(img_dir).mkdir(exist_ok=True)
        

        show_img(input['image'][0], inter_dst=2).save(f'{img_dir}/P{i1}-{i2}.png')
        print('saved image to', f'{img_dir}/P{i1}-{i2}.png')
        # # exit()
        # print('data shape', input['image'].shape)
        # save to npy
        # np.save('/home/hynx/regis/dongzhong/nifti/liver/P{}-{}.npy'.format(i1, i2), input['image'][0])

    def br_niigze(i,j):
        reader = sitk.ImageSeriesReader()
        dc_name=str(j)
        
        # br
        i1 = str(i)[str(i).find('P')+1:][:1]
        i2 = str(j)[str(j).find('P')+3:][:1]
        i1 = int(i1)
        i2 = int(i2)
        print(i1, i2)
        print('reading', dc_name)
        
        if True:
            nii_name = '/home/hynx/regis/zdongzhong/nifti/brain/original/P{}-{}.nii.gz'.format(i1, i2)
            pa(nii_name).parent.mkdir(parents=True, exist_ok=True)
            if pa(nii_name).exists():
                # rm
                pa(nii_name).unlink()

            dicom_names = reader.GetGDCMSeriesFileNames(dc_name)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()

            # # Added a call to PermuteAxes to change the axes of the data
            image = sitk.PermuteAxes(image, [2, 1, 0])
            print('data spacing', image.GetSpacing())
            print('data shape', image.GetSize())
            sitk.WriteImage(image, nii_name)

        input = br_val_transforms({'image': nii_name})
        # read image affine after transform
        print(input['image'].shape)
        print(input['image_meta_dict']['affine'])
        # spacing
        print(input['image_meta_dict']['pixdim'])
        # get image orientation
        import pprint
        # pprint.pprint(input['image_meta_dict'])
        img_dir = '/home/hynx/regis/zdongzhong/images/brain_nii'
        pa(img_dir).mkdir(exist_ok=True)
        show_img(input['image'], inter_dst=5).save(f'{img_dir}/P{i1}-{i2}.png')
        print('saved image to', f'{img_dir}/P{i1}-{i2}.png')
        print()

            
    root_liver = '/home/hynx/regis/zdongzhong/lm'
    root_brain = '/home/hynx/regis/zdongzhong/dcom/brain'
    brain_sen = {
        "1-1":1,
        "1-2":1,
        "2-1":2,
        "2-2":1,
        "3-1":1,
        "3-2":0,
        "4-1":2,
        "4-2":1,
        "5-1":0,
        "5-2":2,
    }
    for i1, i, in enumerate(sorted(pa(root_brain).glob('P*'))):
        # for i2,j in enumerate(sorted(pa(i).glob('ST*'))):
        se_i = brain_sen[str(i)[-3:]]
        print(i, se_i)
        for i2,j in enumerate(sorted(pa(i).glob(f'**/SE{se_i}'))):
            br_niigze(i,j)
#%% extract brain mask
import nibabel as nib
import numpy as np
from skimage.measure import label
import ants
from pathlib import Path as pa
from monai.visualize import matshow3d
import subprocess
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

def bet(src_path, dst_path, frac="0.5"):
    command = ["bet", src_path, dst_path, "-R", "-f", frac, "-g", "0"]
    print(" ".join(command))
    subprocess.call(command)
    return

def strip_skull(src_path, dest_path):
    pa(dest_path).parent.mkdir(parents=True, exist_ok=True)
    print('stripping skull', src_path, dest_path)
    bet(src_path, dest_path.replace(".nii.gz", ""))

    img = nib.load(src_path).get_fdata()
    print('img shape', img.shape)
    str_img = nib.load(dest_path).get_fdata()
    fig, (ax1, ax2) = plt.subplots(1, 2)
    matshow3d(img, fig=ax1, cmap='gray', margin=1)
    matshow3d(str_img, fig=ax2, cmap='gray', margin=1)
    img_dir = '/home/hynx/regis/zdongzhong/images/brain_strp_skull_nii'
    pa(img_dir).mkdir(exist_ok=True)
    save_name = pa(img_dir)/ pa(src_path).with_suffix('.png').name
    plt.savefig(save_name)
    plt.close()

brain_nii_path = '/home/hynx/regis/zdongzhong/nifti/brain/original'
srcs, dests = [], []
for im in sorted(pa(brain_nii_path).glob('P*.nii.gz')):
    srcs.append(str(im))
    dests.append(str(im.parent/ 'striped'/im.name))

# strip_skull(srcs[0], dests[0])
pool = Pool(4)
pool.starmap(strip_skull, zip(srcs, dests))
#%% to nnunet segment
from pathlib import Path as pa
ddir = '/home/hynx/regis/zdongzhong/nifti/brain/original/striped'
for i in sorted(pa(ddir).glob('P*.nii.gz')):
    i.rename(str(i).replace('.nii.gz', '_0000.nii.gz'))
    print(i, i.name.replace('.nii.gz', '_0000.nii.gz'))

#  nnUNet_predict -i /home/hynx/regis/zdongzhong/nifti/brain/original/striped -o /home/hynx/regis/zdongzhong/nifti/brain/seg_unet -t 83 -f 0
#  nnUNet_predict -i /home/hynx/regis/zdongzhong/nifti/brain/original/striped -o /home/hynx/regis/zdongzhong/nifti/brain/seg_unet_bg -t 83 -f 2 -tr nnUNetTrainerV2BraTSRegionsBG
#%% rename cases
from pathlib import Path as pa
for i in pa('/home/hynx/regis/zdongzhong/nifti/brain/original/striped').glob('*.nii.gz'):
    # copy to parent
    print(i)
    new_name = i.name.replace('_0000','')
    # copy and rename
    import shutil
    shutil.copy(i, '/home/hynx/regis/zdongzhong/nifti/brain/striped/'+new_name)

#%% visualize
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from monai.visualize import matshow3d
from tools.utils import *
d_path = '/home/hynx/regis/zdongzhong/nifti/brain/striped/'
s_path = '/home/hynx/regis/zdongzhong/nifti/brain/seg_unet/'

ims = []
for d, s in zip(sorted(pa(d_path).glob('P*.nii.gz')), sorted(pa(s_path).glob('P*.nii.gz'))):
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # matshow3d(nib.load(d_path).get_fdata(), fig=ax1, cmap='gray', margin=1)
    # matshow3d(nib.load(s_path).get_fdata()>1, fig=ax2, cmap='gray', margin=1)
    im = combo_imgs(nib.load(d).get_fdata(), 
            (nib.load(s).get_fdata()).astype(float),
            idst=1)
    ims.append(im)
#%%
# ims[0]

#%% get h5
import nibabel as nib
import torch.nn.functional as F
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    Orientationd,
    EnsureTyped,
    CropForegroundd,
    NormalizeIntensityd,
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
    # pad: tdl-bur
    if isinstance(pad, int):
        pad = [pad, pad, pad, pad, pad, pad]
    bbox = bbox.copy()
    # increase bbox by pad
    bbox[0] = max(bbox[0] - pad[0], 0)
    bbox[1] = max(bbox[1] - pad[1], 0)
    bbox[2] = max(bbox[2] - pad[2], 0)
    bbox[3] += pad[3]
    bbox[4] += pad[4]
    bbox[5] += pad[5]
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
    if 'volume' in f[group]:
        del f[group]['volume']
    f[group].create_dataset('volume', data=data, dtype='uint8')
    if 'segmentation' in f[group]:
        del f[group]['segmentation']
    f[group].create_dataset('segmentation', data=seg, dtype='uint8')
# for 1-4: just 

li_val_transforms = Compose(
    [
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        # Orientationd(keys=["image", "label"], axcodes="RAS"),
        EnsureTyped(keys=["image","label"], device='cpu', track_meta=True),
    ]
)
def li2h5():
    data_name = pa(t1_seg)
    pad =5
    if '4-0' in str(data_name):
        pad = (150,5,5,5,5,5)
    # else: continue
    print(t1, t1_seg)
    t1 = nib.load(t1).get_fdata()
    t1_seg = nib.load(t1_seg).get_fdata()
    t1 = torch.from_numpy(t1)
    t1_seg = torch.from_numpy(t1_seg)
    # choose largest connected component
    l_seg = getLargestCC(t1_seg>0)
    bbox = bbox_seg(l_seg)
    t1_seg[~l_seg] = 0
    t1_seg = crop(t1_seg, bbox, pad)
    t1 = crop(t1, bbox, pad)
    print(t1.shape, t1_seg.shape)
    # if 'P5' not in str(data_name):
    t1 = orient(t1)
    t1_seg = orient(t1_seg)
    t1 = F.interpolate(t1.unsqueeze(0).unsqueeze(0), size=(128,128,128), mode='trilinear').squeeze(0).squeeze(0)
    t1_seg = F.interpolate(t1_seg.unsqueeze(0).unsqueeze(0), size=(128,128,128), mode='nearest').squeeze(0).squeeze(0)
    print(t1.shape, t1_seg.shape)
    
    img_dir = '/home/hynx/regis/zdongzhong/images/liver/'
    pa(img_dir).mkdir(exist_ok=True)
    show_img(t1, inter_dst=1).save('{}{}_img.png'.format(img_dir, data_name.name[:-7]))
    t1 = F.interpolate(t1.unsqueeze(0).unsqueeze(0), size=(128,128,128), mode='trilinear').squeeze(0).squeeze(0)
    show_img(t1, inter_dst=5).save('{}{}_img_128.png'.format(img_dir, data_name.name[:-7]))
    print('save img to ', '{}{}_img.png'.format(img_dir, data_name.name[:-7]))

    dct = li_val_transforms({"image": t1, "label": t1_seg})
    dt1 = dct['image']
    print(dt1.shape, )
    show_img(t1, inter_dst=5).save('{}{}_img_128h5.png'.format(img_dir, data_name.name[:-7]))
    # combo_imgs(dt1, dt1_seg, axis=0, idst=5).save('/home/hynx/regis/zdongzhong/images/liver/{}_combo.png'.format(data_name.name[:-7]))
    # seg_on_img = draw_seg_on_vol(dt1, dt1_seg.long(), inter_dst=5, to_onehot=True)
    # show_img(seg_on_img, norm=False, inter_dst=1).save('/home/hynx/regis/zdongzhong/images/liver/{}.png'.format(data_name.name[:-7]))
    # print('save to ', '/home/hynx/regis/zdongzhong/images/liver/{}.png'.format(data_name.name[:-7]))

    # save h5
    write_h5(h5file, data_name.name[1:4], dt1, dt1_seg)

br_val_transforms = Compose(
    [
        EnsureTyped(keys=["image"]),
        # Orientationd(keys=["image"], axcodes="RAS"),
        # Spacingd(
        #     keys=["image"],
        #     pixdim=(3., 1.0, 1.0),
        #     mode=("bilinear"),
        # ),
        # CropForegroundd(keys=["image"], source_key="label"),
        # NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ]
)
def br2h5(t1, t1_seg):
    data_name = pa(t1_seg)
    print(t1, t1_seg)
    
    ## read data
    t1 = nib.load(t1).get_fdata()
    t1_seg = nib.load(t1_seg).get_fdata()
    t1 = torch.from_numpy(t1)
    t1_seg = torch.from_numpy(t1_seg)

    ### get seg
    # get foreground from img
    # t1_seg[t1_seg>0] = t1_seg[t1_seg>0]+1
    t1_seg[t1_seg>1] = 2
    t1_seg[(t1>t1.min()) & (t1_seg<=0)] = 1
    # choose largest connected component
    l_seg = getLargestCC(t1_seg>0)
    print('original_shape', t1.shape, l_seg.shape, t1_seg.shape)
    bbox = bbox_seg(l_seg)
    t1_seg[~torch.from_numpy(l_seg)] = 0
    if '3-2' in str(data_name):
        t1[t1>1000] = 1001
        t1_seg[t1>1000][:8] = 0
        t1_seg[t1>1000][-3:] = 0
    # background value
    mid_slice = np.s_[t1.shape[0]//2: t1.shape[0]//2 + 1]
    def norm(x, seg):
        # x[seg>0] = (x[seg>0]-x[seg>0].mean())/(x[seg>0].std()+1e-8)
        return x[seg>0].mean(), x[seg>0].std()
    mean, std = norm(t1[mid_slice], t1_seg[mid_slice])
    t1[t1_seg>0] = (t1[t1_seg>0]-mean)/(std+1e-8)
    t1[t1_seg==0] = t1[t1_seg>0].min()-0.1
    # crop
    pad =20
    t1_seg = crop(t1_seg, bbox, pad)
    t1 = crop(t1, bbox, pad)
    print('after_crop', t1.shape, t1_seg.shape)

    # get orient
    # t1 = t1.permute(2,1,0)
    # t1_seg = t1_seg.permute(2,1,0)
    img_dir = '/home/hynx/regis/zdongzhong/images/brain/'
    pa(img_dir).mkdir(exist_ok=True)
    show_img(t1, inter_dst=1).save('{}{}_original_img.png'.format(img_dir, data_name.name[:-7]))
    print('save img to ', '{}{}_original_img.png'.format(img_dir, data_name.name[:-7]))

    t1 = F.interpolate(t1.unsqueeze(0).unsqueeze(0), size=(128,128,128), mode='trilinear').squeeze(0).squeeze(0)
    t1_seg = F.interpolate(t1_seg.unsqueeze(0).unsqueeze(0), size=(128,128,128), mode='nearest').squeeze(0).squeeze(0)
    print(t1.shape, t1_seg.shape)  
    ## to 128x128x128
    pa(img_dir).mkdir(exist_ok=True)
    # show_img(t1, inter_dst=5).save('{}{}_img_128.png'.format(img_dir, data_name.name[:-7]))

    ## preprocess
    dct = br_val_transforms({"image": t1, "label": t1_seg})
    dt1 = dct['image']
    dt1_seg = dct['label']
    print(dt1.shape, )
    show_img(t1, inter_dst=5).save('{}{}_128h5.png'.format(img_dir, data_name.name[:-7]))
    combo_imgs(dt1, dt1_seg, axis=0, idst=5).save('/home/hynx/regis/zdongzhong/images/brain/{}_combo.png'.format(data_name.name[:-7]))

    seg_on_img = draw_seg_on_vol(dt1, dt1_seg.round().long(), inter_dst=5, to_onehot=True, colors=['red', 'green', 'blue', 'yellow'])
    show_img(seg_on_img, norm=False, inter_dst=1).save('/home/hynx/regis/zdongzhong/images/brain/{}.png'.format(data_name.name[:-7]))
    print('save to ', '/home/hynx/regis/zdongzhong/images/brain/{}.png'.format(data_name.name[:-7]))

    # save h5
    write_h5(h5file, data_name.name[1:4], dt1, dt1_seg)

import h5py
h5file = '/home/hynx/regis/zdongzhong/nifti/brain_test_23.h5'
# if not pa(h5file).exists():
h5file = h5py.File(h5file, 'w')
for t1, t1_seg in zip(
    sorted(pa('/home/hynx/regis/zdongzhong/nifti/brain/striped').glob('*.nii.gz')),
    sorted(pa('/home/hynx/regis/zdongzhong/nifti/brain/seg_unet').glob('*.nii.gz'))
):
    br2h5(t1, t1_seg)
        
h5file.close()

#%%
