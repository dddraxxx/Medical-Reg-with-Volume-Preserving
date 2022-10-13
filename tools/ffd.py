# %%
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
import pickle as pkl
import numpy as np
from pathlib import Path as pa
import pygem as pg

from adet.utils.visualize_niigz import *

def show_img(res):
    import torchvision.transforms as T
    res = tt(res)
    if res.ndim>=3:
        return T.ToPILImage()(visulize_3d(res))
    # normalize res
    res = (res-res.min())/(res.max()-res.min())
    return T.ToPILImage()(res)

def plt_grid3d(pts, ax, **kwargs):
    l = np.round(len(pts)**(1/3)).astype(int)
    x, y, z = pts.T.reshape(-1, l, l, l)
    grid1 = np.stack((x,y,z), axis=-1)
    ax.add_collection3d(Line3DCollection(grid1.reshape(-1, l, 3), **kwargs))
    ax.add_collection3d(Line3DCollection(grid1.transpose(0,2,1,3).reshape(-1, l, 3),  **kwargs))
    ax.add_collection3d(Line3DCollection(grid1.transpose(1,2,0,3).reshape(-1, l, 3),  **kwargs))

from PIL import Image
def combine_pil_img(*ims):
    widths, heights = zip(*(i.size for i in ims))
    total_width = sum(widths) + 5*(len(ims)-1)
    total_height = max(heights)
    im = Image.new('RGB', (total_width, total_height), color='white')
    for i, im_ in enumerate(ims):
        im.paste(im_, (sum(widths[:i])+5*i, 0))
    return im

def save_niigz(arr, path):
    import itk
    img = itk.GetImageFromArray(arr)
    itk.imwrite(img, path)

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
# find a location to paste tumour
def find_loc_to_paste_tumor(seg1, kseg2, bbox):
    '''
    1: organ, 2:tumor'''
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
    pts = pts[::10]
    pts = np.random.permutation(pts)
    if len(pts):
        for i in range(len(pts)):
            pt = pts[i]
            s1 = seg1[pt[0]:pt[0]+box_length[0], pt[1]:pt[1]+box_length[1], pt[2]:pt[2]+box_length[2]]==1
            s2 = kseg2[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], bbox[0][2]:bbox[1][2]]==2
            if (s1&s2 == s2).all():
                return np.array([pt, pt+box_length])
    raise ValueError("cannot find a location to paste tumor")

def find_largest_component(seg):
    from monai.transforms import KeepLargestConnectedComponent
    k = KeepLargestConnectedComponent(2)
    kseg = np.maximum(k(seg[None].copy()), seg>0)
    return kseg

def find_bbox(seg2):
    from monai.transforms.utils import generate_spatial_bounding_box
    kseg2 = find_largest_component(seg2)
    bbox = generate_spatial_bounding_box(kseg2, lambda x:x==2)
    kseg2 = kseg2[0]
    o_bbox = np.array(bbox)
    f_bbox = find_loc_to_paste_tumor(seg1, kseg2, o_bbox)
    print("origin bbox", o_bbox)
    box_length = f_bbox[1]-f_bbox[0]
    ext_l = np.random.rand(3)*(box_length/2)
    print('box length', box_length, '\nextenede length', ext_l)
    bbox = np.stack([np.maximum(f_bbox[0]-ext_l, 0), np.minimum(f_bbox[1]+ext_l, 128-1)], axis=0)
    # print('extended bbox', bbox[0], bbox[1])
    # print("find bbox", f_bbox)
    # print("new center", np.array([c_x, c_y, c_z]) - o_bbox[0] + f_bbox[0])
    return kseg2, o_bbox, f_bbox, bbox

def draw_3dbox(res, l, r, val=255):
    res = res.copy()   
    res[l[0]:r[0], l[1], l[2]:r[2]] = val
    res[l[0]:r[0], r[1], l[2]:r[2]] = val
    res[l[0]:r[0], l[1]:r[1], l[2]] = val
    res[l[0]:r[0], l[1]:r[1], r[2]] = val
    return res
    
from scipy.ndimage import map_coordinates
# Small disp: 0.2, 0.4
# Large disp: 0.4, 0.4
def presure_ffd(o_bbox, f_bbox, bbox, kseg2, fct_low=0.4, fct_range=0.4):
    l = 8
    ffd = pg.FFD([l, l, l])
    ffd.box_origin = bbox[0]
    ffd.box_length = bbox[1]-bbox[0]
    control_pts = ffd.control_points(False).reshape(l,l,l,3)[1:-1,1:-1,1:-1]
    # calculate tumor map for each control point
    paste_seg2 = np.zeros_like(kseg2)
    paste_seg2[f_bbox[0][0]:f_bbox[1][0], f_bbox[0][1]:f_bbox[1][1], f_bbox[0][2]:f_bbox[1][2]] = kseg2[o_bbox[0][0]:o_bbox[1][0], o_bbox[0][1]:o_bbox[1][1], o_bbox[0][2]:o_bbox[1][2]]
    seg_pts = map_coordinates(paste_seg2==2, control_pts.reshape(-1,3).T, order=0, mode='nearest')
    seg_pts = seg_pts.reshape(*control_pts.shape[:3])
    # calculate the pressure
    dsp_matrix = control_pts.reshape(-1, 3) - control_pts.reshape(-1, 3)[:, None]
    dsp_matrix /= ffd.box_length
    # diagonal entries are the same point, and no need to calculate the distance
    dst_matrix = (dsp_matrix**2).sum(axis=-1)
    np.fill_diagonal(dst_matrix, 1)
    fct = np.random.rand()*fct_range+fct_low
    print('factor', fct)
    prs_matrix = fct*dsp_matrix/dst_matrix[...,None]
    prs_pts = (prs_matrix*seg_pts.reshape(-1,1)).sum(axis=1)
    dx, dy, dz = prs_pts.reshape(*control_pts.shape).transpose(-1,0,1,2)/((l-1)**3)
    # dx, dy, dz = prs_pts.reshape(*control_pts.shape).transpose(-1,0,1,2)/(l-1)/ffd.box_length[:,None,None,None]
    ffd.array_mu_x[1:-1,1:-1,1:-1] = dx
    ffd.array_mu_y[1:-1,1:-1,1:-1] = dy
    ffd.array_mu_z[1:-1,1:-1,1:-1] = dz

    ## %% plot deform
    # %matplotlib widget
    # fig = plt.figure(figsize=(4, 4))
    # ax = Axes3D(fig)
    # ax.scatter(*control_pts[seg_pts].T, s=20, c='green')
    # # ax.scatter(*control_pts[~seg_pts].T, s=20)
    # ax.scatter(*ffd.control_points().T, s=5, c='red')
    # # plt_grid3d(ffd.control_points(False), ax, colors='b', linewidths=1)
    # plt_grid3d(ffd.control_points(), ax, colors='r', linewidths=0.5)
    # # show x,y,z axis
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # ax.view_init(3, 90)
    return ffd, paste_seg2

def do_ffd(ffd, img, seg=None):
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

##%% paste tumor
def roll_pad(data, shift, padding=0):
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
    global dist_pts, weight_kseg2, dst_to_weight
    dist_pts = distance_transform_edt(kseg2!=2) - distance_transform_edt(kseg2==2)
    weight_kseg2 = np.zeros_like(kseg2, dtype=np.float32)
    dst_to_weight = lambda x, l: 0.5-x/2/l
    idx = abs(dist_pts)<feather_len
    weight_kseg2[idx] = dst_to_weight(dist_pts[idx], feather_len)
    weight_kseg2[dist_pts<=-feather_len] = 1
    # feathering area should be inside organ
    inside_organ = kseg2>0
    weight_kseg2[~inside_organ] = 0
    return weight_kseg2

# distance map
def dm(kseg2, feather_len):
    global dist_pts, weight_kseg2, dst_to_weight
    dist_pts = distance_transform_edt(kseg2!=2)
    weight_kseg2 = np.zeros_like(kseg2, dtype=np.float32)
    dst_to_weight = lambda x, l: 1-x/l
    idx = dist_pts<feather_len
    weight_kseg2[idx] = dst_to_weight(dist_pts[idx], feather_len)
    inside_organ = kseg2>0
    weight_kseg2[~inside_organ] = 0
    return weight_kseg2

def direct_paste(res, img2, kseg2, f_bbox, o_bbox):
    weight_kseg2 = kseg2.copy()
    weight_seg1 = roll_pad(1-weight_kseg2, f_bbox[0]-o_bbox[0], padding=1)
    roll_img2 = roll_pad(img2, f_bbox[0]-o_bbox[0], padding=0)
    res1 = res*weight_seg1 + roll_img2*(1-weight_seg1)
    # res1 = res.copy()
    # res1[paste_seg2==2] = img2[kseg2==2]
    # direct paste tumor with borders
    # res1 = res.copy()
    # roll_dist = roll_pad(dist_pts, f_bbox[0]-o_bbox[0], padding=feather_len+1)
    # res1[roll_dist<feather_len]=img2[dist_pts<feather_len]
    return res1

def feathering_paste(feather_len, res, img2, kseg2, f_bbox, o_bbox):
    weight_kseg2 = sdm(kseg2, feather_len)
    weight_seg1 = roll_pad(1-weight_kseg2, f_bbox[0]-o_bbox[0], padding=1)
    roll_img2 = roll_pad(img2, f_bbox[0]-o_bbox[0], padding=0)
    res1 = res*weight_seg1 + roll_img2*(1-weight_seg1)
    return res1


## %% part 2 of the deform: random deformation
def random_ffd(l=5):
    ffd = pg.FFD([l, l, l])
    ffd.box_length = np.full(3, 128-1)
    ffd.box_origin = np.zeros(3)
    # get random field
    x_field, y_field, z_field = (np.random.rand(3, l, l, l)-0.5) /(l-1)
    ffd.array_mu_x += x_field
    ffd.array_mu_y += y_field
    ffd.array_mu_z += z_field
    return ffd

def plt_ffd(ffd):
    fig = plt.figure(figsize=(4, 4))
    ax = Axes3D(fig)
    ax.scatter(*ffd.control_points(False).T, s=10)
    ax.scatter(*ffd.control_points().T, s=10, c='red')
    # plt_grid3d(ffd.control_points(False), ax, colors='blue')
    plt_grid3d(ffd.control_points(), ax, colors='red')
    plt.show()

def ffd_params(ffd):
    return np.array([*ffd.n_control_points, *ffd.box_origin, *ffd.box_length, *ffd.array_mu_x.flatten(), *ffd.array_mu_y.flatten(), *ffd.array_mu_z.flatten()])

#%%
if __name__=='__main__':
    # use h5
    import h5py
    fname = '/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits.h5'
    dct = h5py.File(fname, 'r')

    # tumor sorted idx
    # [ 87  89 105 106  32  34  91 114 115  38  41  47 119 ||  83 127  25  12   5
    #  112  15 125  59  67  20  65  95  24  92  58  73  54   3  63  14  42  86
    #  121  55 111 107  69  61  62  57 126  81  85 120  43  66  11  99  68  45
    #   75  50  53   0   2  19  77  60  18 102  31  30  35  10  96   8   9  29
    #    7   1  37  79   6  21  49  13  17  78  94  72  23  26 109  22 103  52
    #  113  36  82 122  48 110  74 124  70 128  40  46  27 101  80  93  88  90
    #   44 123  28  39  84  64 104  16  76  51  97  71  98  56 116 118 117 129
    #   33 100 108 130   4]
    # no tumor: ['87',  '89', '105', '106',  '32',  '34',  '91', '114', '115',  '38',  '41',  '47', '119',]
    # no tumor instance
    id1s = ['106','119','38']
    # with tumor
    id2s = ['104',  '51', '118', '129']
    # below tumor too large 
    # id2s = ['129', "33", "100", "108", "130", "4"]
    write_h5 = False
    validate = False
    if write_h5:
        h5_writer = h5py.File('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/lits_deform_fL.h5', 'w')
    img_dir = '/home/hynx/regis/zFFD/'
    #%%
    for id1 in id1s:
        for id2 in id2s:
            print('\ndeforming organ {} for tumor {}'.format(id1, id2))
            img1 = dct[id1]['volume'][...]
            img2 = dct[id2]['volume'][...]
            seg1 = dct[id1]['segmentation'][...]
            seg2 = dct[id2]['segmentation'][...]

            combine_pil_img(show_img(seg1), show_img(seg2), show_img(find_largest_component(seg2)[0])).save(img_dir+'img/{}-{}_seg.png'.format(id1, id2))
            combine_pil_img(show_img(img1), show_img(img2)).save(img_dir+'img/{}-{}_img.png'.format(id1, id2))

            try:
                kseg2, o_bbox, f_bbox, bbox = find_bbox(seg2)
            except ValueError as e:
                print(e)
                continue
            p_ffd, paste_seg2 = presure_ffd(o_bbox, f_bbox, bbox, seg2)
            res, res_seg = do_ffd(p_ffd, img1, seg1)
            feather_len = 5
            # paste tumor
            # res1 = direct_paste()
            res1 = feathering_paste(feather_len, res, img2, kseg2, f_bbox, o_bbox)
            res_seg[paste_seg2==2]=2
            # just for validation
            # res1 = res

            # dres = draw_3dbox(res1, f_bbox[0], f_bbox[1])
            # dimg2 = draw_3dbox(img2, o_bbox[0], o_bbox[1])
            # combine_pil_img(show_img(dres), show_img(dimg2), show_img(res1))
            ## %%
            # show_img(distance_transform_edt(kseg2!=2) - distance_transform_edt(kseg2==2))
            # show_img(sdm(kseg2, feather_len))
            # show_img(weight_kseg2*img2)
            #%%
            r_ffd = random_ffd() # l=5 the default settings
            # plt_ffd(ffd)
            res2, res2_seg = do_ffd(r_ffd, res1, res_seg)

            mesh_pts = np.mgrid[0:128, 0:128, 0:128].reshape(3, -1).T.astype(float)
            # below the composition of flow
            ffd_pts = p_ffd(r_ffd(mesh_pts))
            ffd_pts = ffd_pts.reshape(128, 128, 128, 3)
            # validate if the composite flow is right
            if validate:
                val = map_coordinates(img1, ffd_pts.transpose(-1,0,1,2), order=3, mode='constant', cval=0)
                valseg = map_coordinates(seg1, ffd_pts.transpose(-1,0,1,2), order=0, mode='constant', cval=0)
                print('Validate if the composite flow is right:')
                print(abs(val.astype(int)-res2.astype(int)).max(), abs(res2).max(), img1.max())
                print(np.histogram(abs(val.astype(int)-res2.astype(int)), 10))
                print(abs(valseg.astype(int)-res2_seg.astype(int)).max(), abs(res2_seg).max(), seg1.max())
                print(np.histogram(abs(valseg.astype(int)-res2_seg.astype(int)), 10))
                show_img(abs(valseg.astype(int)-res2_seg.astype(int))).save('s.jpg')
                # save img
                im = combine_pil_img(show_img(val), show_img(valseg))
                im.save(img_dir+'val/{}_{}.png'.format(id1, id2))

            im = combine_pil_img(show_img(img1), show_img(np.clip(res2,0,255)), show_img(res2_seg))
            im.save(img_dir+f'./deform/{id1}_{id2}.png')

            # save img1, res2, ffd_gt, and segmentation of res2 to h5 file
            # create group if not exist
            if write_h5:
                if id1 not in h5_writer:
                    h5_writer.create_group(id1)
                    h5_writer[id1].create_dataset('volume', data=np.clip(img1,0,255).astype(np.uint8))
                    h5_writer[id1].create_dataset('segmentation', data=seg1.astype(np.uint8))
                if 'id2' not in h5_writer[id1]:
                    h5_writer[id1].create_group('id2')
                h5_writer[id1]['id2'].create_group(id2)
                h5_writer[id1]['id2'][id2].create_dataset('volume', data=np.clip(res2,0,255).astype(np.uint8))
                h5_writer[id1]['id2'][id2].create_dataset('segmentation', data=res2_seg.astype(np.uint8))
                h5_writer[id1]['id2'][id2].create_dataset('ffd_gt', data=ffd_pts.astype(np.float32))
                # save ffd in h5
                ffd_obj = np.concatenate([ffd_params(p_ffd), ffd_params(r_ffd)])
                h5_writer[id1]['id2'][id2].create_dataset('ffd', data=ffd_obj)

    if write_h5: h5_writer.close()