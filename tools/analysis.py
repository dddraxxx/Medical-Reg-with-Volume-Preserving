#%%
from pprint import pprint
import numpy as np
import pickle as pkl
import numpy as np
import os
from pathlib import Path as pa
import sys
sys.path.append('..')
sys.path.append('.')
from networks.spatial_transformer import SpatialTransform
from utils import draw_seg_on_vol, show_img, visualize_3d
import torch
from thres_ffd import FFD_GT


def eP2P(flow, gt, mask):
    eP2P = abs(flow) - abs(gt)
    # no tumor in
    eP2P = eP2P[~mask]
    # print(eP2P.shape)
    err = ((eP2P**2).sum(axis=-1)).mean()
    return err

def cal_warped_seg(agg_flows, seg2):
    st = SpatialTransform(seg2[0,0].shape)
    w_segs = st(seg2, agg_flows, mode='nearest')
    return w_segs

def cal_single_warped(flow, img):
    st = SpatialTransform(img.shape)
    w_img = st(img[None,None], flow[None], mode='bilinear')
    return w_img[0,0]
#%%
fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Sep27_145203_normal_lits_d_.pkl'
mode = 'normal'
# fname = '/home/hynx/regis/recursive-cascaded-networks/evaluations/Oct03_210121_masked_lits_d_.pkl'
# mode = 'mask'

dct = pkl.load(open(fname, 'rb'))
print(dct.keys())
print(dct['dice_liver'])
#%%
idx = 0
errs = []

h5_name = '/home/hynx/regis/recursive-cascaded-networks/datasets/lits_deform_L.h5'
reader = FFD_GT(h5_name)

save_im = False
for idx in range(len(dct['id1'])):
    agg_flow = dct['agg_flows'][idx][-1].permute(1,2,3,0).numpy()
    
    id1 = dct['id1'][idx][7:]
    id2 = dct['id2'][idx]
    print('analyzing', id1, id2)
    reader.set_to(id1, id2);
    gt = reader.ffd_gt
    gt_flow = gt - np.mgrid[0:gt.shape[0], 0:gt.shape[1], 0:gt.shape[2]].astype(np.float32).transpose(1,2,3,0)
    seg2 = reader.seg2
    seg1 = reader.seg1

    w_segs= cal_warped_seg(dct['agg_flows'][idx], torch.tensor(seg2)[None,None].repeat(4,1,1,1,1).float())
    for i, s in enumerate([seg2, *w_segs, seg1]):
        print('{}th layer| organ size {}| tumor size {}'.format(i, (s>0.5).sum(), (s>1.5).sum()))

    masks = [seg2>1.5]
    thres = np.percentile(reader.disp, [85, 95, 99])
    masks = masks + [np.logical_or(seg2>1.5, m) for m in reader.get_masks(thres)]
    err = [eP2P(agg_flow, gt_flow, m) for m in masks]
    print('{} to {} flow end to end error:'.format(id1, id2), err)
    errs.append(err)

    # save image
    if save_im:
        img1 = reader[id1]['volume'][...]
        img2 = reader[id1]['id2'][id2]['volume'][...]
        img_dir = pa('../images/{}/{}_{}'.format(mode, id1, id2))
        img_dir.mkdir(parents=True, exist_ok=True)
        show_img(img1).save(img_dir/'img1.png')
        # draw tumor seg to img2
        s_img2 = draw_seg_on_vol(img2[None], (seg2==2)[None], colors=[(250,0,0)])
        show_img(s_img2).save(img_dir/'img2.png')
        # save warped
        warped = cal_single_warped(dct['agg_flows'][idx][-1], torch.tensor(img2).float())
        s_warped = draw_seg_on_vol(warped[None], (w_segs[-1]==2), colors=[(250,0,0)])
        show_img(s_warped).save(img_dir/'s_warped.png')
        show_img(warped).save(img_dir/'warped.png')
        print(img_dir/'warped.png')
    
    print()
print(errs)
print(mode, 'mean pnp error:', np.mean(errs, axis=0), 'std:', np.std(errs, axis=0))

#%%
# np.histogram(reader.disp, bins=5)    
np.percentile(reader.disp, [85, 95, 99])
# %%
from utils import combine_pil_img
combine_pil_img(*[show_img(im) for im in masks])
#%% close reader
reader.close()