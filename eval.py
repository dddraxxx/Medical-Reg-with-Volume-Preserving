import torchvision.transforms as T
from metrics.losses import *
from tools.flow_display import flow_to_image
import time
from pathlib import Path as pa
import argparse
from genericpath import isfile
import os
import pickle
import json
import re
from matplotlib import pyplot as plt
import numpy as np
from metrics.losses import dice_jaccard, find_surf
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch
from metrics.surface_distance import *
from networks.recursive_cascade_networks import RecursiveCascadeNetwork
from data_util.dataset import Data, Split
from tools.utils import *
from run_utils import build_precompute, read_cfg
from tools.visualization import *

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Specifies a previous checkpoint to load')
parser.add_argument('-g', '--gpu', type=str, default='0',
                    help='Specifies gpu device(s)')
parser.add_argument('-d', '--dataset', type=str, default=None,
                    help='Specifies a data config')
parser.add_argument('-v', '--val_subset', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=4, help='Size of minibatch')
parser.add_argument('-s','--save_pkl', action='store_true', help='Save the results as a pkl file')
parser.add_argument('-re','--reverse', action='store_true', help='If save reverse flow in pkl file')
parser.add_argument('-tl','--test_large', action='store_true', help='If test on data with small tumor')
parser.add_argument('-tb','--test_boundary', action='store_true', help='If test on data with tumor close to organ boundary')
parser.add_argument('-lm', '--lmd', action='store_true', help='If test landmark locations')
parser.add_argument('--lmk_json', type=str, default='/home/hynx/regis/recursive-cascaded-networks/landmark_json/lits17_landmark.json', help='landmark for eval files')
# parser.add_argument('-m', '--masked', action='store_true', help='If model need masks')
parser.add_argument('-lm_r', '--lmk_radius', type=int, default=10, help='affected landmark within radius')
parser.add_argument('-vl', '--visual_lmk', action='store_true', help='If visualize landmark')
parser.add_argument('-rd', '--region_dice', default=True, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='If calculate dice for each region')
parser.add_argument('-sd', '--surf_dist', default=False, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='If calculate dist for each surface')
parser.add_argument('-only_vis', '--only_vis_target', action='store_true', help='If only visualize target')
parser.add_argument('--use_ants', action='store_true', help='if use ants to register')
parser.add_argument('--debug', action='store_true', help='if debug')
args = parser.parse_args()
if args.checkpoint == 'normal':
    args.checkpoint = '/home/hynx/regis/recursive-cascaded-networks/logs/liver/VTN/Jan08_180325_normal-vtn'

# if args.gpu:
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# set gpu to the one with most free memory
import subprocess
GPU_ID = subprocess.getoutput('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs')
print('Using GPU', GPU_ID)
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

# resolve checkpoint path to realpath 
args.checkpoint = pa(args.checkpoint).resolve().__str__()

def main():
    # update args with checkpoint args but do not overwrite
    model_path = args.checkpoint
    ckp = args.checkpoint
    cfg_training = read_cfg(model_path)
    args.dataset = cfg_training.dataset
    args.checkpoint = ckp
    # for k, v in cfg.items():
    #     if not hasattr(args, k):
    #         setattr(args, k, v)
    # read config
    with open(args.dataset, 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        image_type = cfg.get('image_type', None)
        segmentation_class_value=cfg.get('segmentation_class_value', {'unknown':1})
    # build dataset
    val_dataset = Data(args.dataset, scheme=args.val_subset or Split.VALID)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=min(8, args.batch_size), shuffle=False)
    # build framework
    model = RecursiveCascadeNetwork(n_cascades=cfg_training.n_cascades, im_size=image_size, base_network=cfg_training.base_network, in_channels=2+bool(cfg_training.masked)).cuda()
    # add checkpoint loading
    from tools.utils import load_model, load_model_from_dir
    if os.path.isdir(args.checkpoint):
        model_path = load_model_from_dir(args.checkpoint, model)
    else:
        load_model(torch.load(model_path), model)
    print("Loading checkpoint from {}".format(model_path))
    
    # parent of model path
    import re
    # "([^\/]*_\d{6}_[^\/]*)"gm
    exp_name = re.search(r"([^\/]*-\d{6}_[^\/]*)", model_path).group(1)
    output_fname = './evaluations/{}_{}_{}.txt'.format(exp_name, args.val_subset or Split.VALID, '' if not args.lmd else 'lm{}'.format(args.lmk_radius))
    print('will save to', output_fname)

    # stage 1 model setup
    if cfg_training.masked in ['soft', 'hard']:
        # suppose the training dataset has the same data type of eval dataset
        data_type = 'liver' if 'liver' in cfg_training.dataset else 'brain'
        cfg_training.data_type = data_type
        build_precompute(model, val_dataset, cfg_training)

    # run val
    model.eval()
    results = {}
    results['id1'], results['id2'], results['dices'] = [], [], []
    if args.save_pkl:
        results['agg_flows'] = []
        results['affine_params'] = []
        if args.reverse:
            results['rev_flow'] = []
    metric_keys = []
    def pick_data(idx, data):
        '''idx: batch-dim mask'''
        for k in data.keys():
            if 'id' in k:
                data[k] = [i for l,i in zip(idx, data[k]) if l]
            else: data[k] = data[k][idx,...]
        return data
    if args.lmd:
        jsn = json.load(open(args.lmk_json, 'r'))
    for iteration, data in tqdm(enumerate(val_loader)):
        seg1, seg2 = data['segmentation1'], data['segmentation2']
        if args.test_large:
            large_idx = (seg2>1.5).sum(dim=(1,2,3,4))/(seg2>0.5).sum(dim=(1,2,3,4))
            large_idx = large_idx ==0
            if not large_idx.any(): continue
            else: data = pick_data(large_idx, data)
        if args.test_boundary:
            seg2_surf = find_surf(seg2, 3)
            seg2_surf_tumor = seg2_surf & (seg2>1.5)
            bound_idx = (seg2_surf_tumor.sum(dim=(1,2,3,4))==0)
            if not bound_idx.any(): continue
            else: data=pick_data(bound_idx, data)
        t0 = time.time()
        seg1, seg2 = data['segmentation1'].float(), data['segmentation2'].float()
                
        fixed, moving = data['voxel1'], data['voxel2']
        id1, id2 = data['id1'], data['id2']
        if args.use_ants:
            import ants
            warps = []
            for i in range(fixed.shape[0]):
                im_fixed = ants.from_numpy(fixed.cpu().numpy()[i,0])
                im_moving = ants.from_numpy(moving.cpu().numpy()[i,0])
                reg = ants.registration(fixed=im_fixed, moving=im_moving, type_of_transform='SyN')
                warped = ants.apply_transforms(fixed=im_fixed, moving=im_moving, transformlist=reg['fwdtransforms'])
                warps.append(warped.numpy())
            warped = np.array(warps)[:,None]
        else:
            with torch.no_grad():
                fixed = fixed.cuda()
                moving = moving.cuda()
                seg2 = seg2.cuda()
                if cfg_training.masked =='seg':
                    moving_ = torch.cat([moving, seg2.float()], dim=1)
                elif cfg_training.masked  in ['soft' , 'mask']:
                    input_seg, compute_mask = model.pre_register(fixed, moving, seg2, training=False, cfg=cfg_training)
                    moving_ = torch.cat([moving, input_seg.float().cuda()], dim=1)
                else:
                    moving_ = moving
                if 'normal' in args.checkpoint and False:
                    # add random noise
                    noise = torch.randn_like(fixed)*0.01
                    print(seg2.shape, fixed.shape)
                    org_mean = (fixed*(seg2>1.5)).sum(dim=(1,2,3,4))/(seg2>1.5).sum(dim=(1,2,3,4))
                    fixed[seg2>1.5] = (org_mean[:,None,None,None,None]+noise)[seg2>1.5]
                    show_img(fixed[0,0]).save('3.jpg')
                warped_, flows, agg_flows, affine_params = model(fixed, moving_, return_affine=True)
                if 'normal' in args.checkpoint:
                    # remove noise
                    # fixed = fixed - noise
                    prev_flow = agg_flows[-1]
                    for i in range(2):
                        warped_, flows, agg_flows, affine_params = model(fixed, warped_[-1], return_affine=True)
                        prev_flow = model.composite_flow(prev_flow, agg_flows[-1])
                    agg_flows[-1] = prev_flow
                # do we need rev flow any more?
                # , return_neg=args.reverse)
                warped = [i[:,:1,...] for i in warped_]
                w_seg2 = model.reconstruction(seg2.float(), agg_flows[-1].float())
        t_infer =  time.time() 
        if args.save_pkl:
            magg_flows = torch.stack(agg_flows).transpose(0,1).detach().cpu()
            if args.reverse:
                re_flow = magg_flows[:, -1]
                magg_flows = magg_flows[:, :-1]
                results['rev_flow'].extend(re_flow)
            results['agg_flows'].extend(magg_flows)
            results['affine_params'].extend(affine_params['theta'].detach().cpu())
        # metrics: landmark
        if args.lmd:
            for ix, ag_flow in enumerate(agg_flows):
                # ag_flow = agg_flows[0].new_zeros(agg_flows[0].shape)
                slc = np.s_[:]
                if 'point1' in data and not (data['point1']==-1).any():
                    lmk1 = data['point1'].squeeze().cuda()
                else:
                    lmk1 : torch.Tensor = ag_flow.new_tensor([jsn[i.split('_')[-1].replace('lits','')][slc] for i in id1]) # n, m, 3
                if 'point2' in data and not (data['point2']==-1).any():
                    lmk2 = data['point2'].squeeze().cuda() 
                else:
                    lmk2 = ag_flow.new_tensor([jsn[i.split('_')[-1]][slc] for i in id2]) # n, m, 3
                # exclude landmarks that is close to tumor
                lmk1_w = lmk1 + torch.stack([torch.stack([ag_flow[j, :][([0,1,2],*lmk1[j,i].long())] \
                    for i in range(lmk1.size(1))]) \
                        for j in range(lmk1.size(0))])
                if args.lmk_radius>0:
                    seg2_tumor = seg2.cuda()>1.5
                    # pick index that is not close to tumor
                    radius = args.lmk_radius
                    points = [] # n,3
                    for z in range(-10,10):
                        for y in range(-10,10):
                            for x in range(-10,10):
                                if x**2+y**2+z**2 <= radius**2:
                                    points.append([x,y,z])
                    points = ag_flow.new_tensor(points).long()
                    l2_x_coordinate = lmk2.long()[:, :, None, 0] + points[:,0] # n,m,10*10*10
                    l2_y_coordinate = lmk2.long()[:, :, None, 1] + points[:,1]
                    l2_z_coordinate = lmk2.long()[:, :, None, 2] + points[:,2]
                    l2_batch_coordinate = torch.arange(lmk2.shape[0])[:, None, None] # n, 1, 1
                    seg2_lmk_neighbor = seg2_tumor[:,0][l2_batch_coordinate, l2_x_coordinate, l2_y_coordinate, l2_z_coordinate] # n,m,10*10*10
                    selected = (seg2_lmk_neighbor.sum(dim=-1)==0) # n,m
                    # show selected
                    # print('landmark selected: {}'.format(selected.sum(-1).tolist()))
                    lmk_err = ((lmk2 - lmk1_w).norm(dim=-1)*selected).sum(-1)/selected.sum(-1) # n,m
                else:
                    lmk_err = (lmk2 - lmk1_w).norm(dim=-1).mean(-1)
                if f'{ix}_lmk_err' not in metric_keys:
                    metric_keys.append(f'{ix}_lmk_err')
                    results[f'{ix}_lmk_err'] = []
                results[f'{ix}_lmk_err'].extend(lmk_err.cpu().numpy())

                # visualize landmarks
            if args.visual_lmk:
                from tools.utils import get_nearest
                flow = agg_flows[-1].squeeze()
                points = torch.meshgrid([torch.arange(flow.shape[2]), torch.arange(flow.shape[3]), torch.arange(flow.shape[4])], indexing='ij')
                points = torch.stack(points).to(flow.device)
                flowed_points = points + flow
                flowed_points = flowed_points.permute(0,2,3,4,1).reshape(args.batch_size,-1,3)
                flow = flow.permute(0,2,3,4,1).reshape(args.batch_size,-1,3)
                flow_lmk2 = get_nearest(flowed_points, lmk2, k=1, picked_points=flow).squeeze().round().long()
                lmk2_w = lmk2 - flow_lmk2
                for ix in range(len(id1)):
                    from tools.visualization import plot_landmarks
                    if not os.path.exists(f'./images/landmarks/{id1[ix]}_fixed.png'):
                        fig, axes = plot_landmarks(fixed[ix,0], lmk1[ix], save_path=f'./images/landmarks/{id1[ix]}_fixed.png')
                    # find the dir that is direct child of logs
                    moving_dir = './images/landmarks/{}'.format(exp_name)
                    # mkdir
                    if not os.path.exists(moving_dir):
                        os.mkdir(moving_dir)
                    # plot_landmarks(fixed[ix,0], lmk1_w[ix], fig=fig, ax=axes, color='yellow', save_path=f'{moving_dir}/{id1[ix]}_{id2[ix]}_fiexd.png')
                    # plot_landmarks(moving[ix,0], lmk2[ix], save_path=f'{moving_dir}/{id1[ix]}_{id2[ix]}_moving.png')
                    plot_landmarks(warped[-1][ix,0], lmk1[ix], save_path=f'{moving_dir}/{id1[ix]}_{id2[ix]}_warped_lmk1.png', size=20, color='red')
                    fig, _ = plot_landmarks(warped[-1][ix,0], lmk2_w[ix], save_path=f'{moving_dir}/{id1[ix]}_{id2[ix]}_warped.png', size=20)
                    if "selected" in locals():
                        # add title for fig
                        fig.suptitle(f'{selected.nonzero().squeeze().tolist()}')
                    # close all figs of plt
                    plt.close('all')

        ### Debug use
        pairs = list(zip(id1, id2))
        # target_pair = ('lits_{}'.format(84), 'lits_{}'.format(40))
        target_pair = ('lits_51', '51')
        # target_pairs = [('lits_{}'.format(51), 'lits_{}'.format(33))]
        # target_pair = ('lits_{}'.format(51), 'yanx_{}'.format(14))
        # target_pair = ('lits_{}'.format(115), 'lits_{}'.format(128))
        # if any([p in pairs for p in target_pairs]):
        if target_pair in pairs:
            pair_id = pairs.index(target_pair)
            pairs_img = [fixed[pair_id,0], moving[pair_id,0], warped[-1][pair_id,0]]
            # get largest component
            from skimage.measure import label
            from skimage.color import label2rgb
            pairs_seg_organ = [seg1[pair_id,0]>0.5, seg2[pair_id,0]>0.5, w_seg2[pair_id,0]>0.5]
            pairs_seg = [seg1[pair_id,0]>1.5, seg2[pair_id,0]>1.5, w_seg2[pair_id,0]>1.5]
            labels = [label(i.cpu()) for i in pairs_seg]
            # warped labels[1]
            labels[2] = model.reconstruction(\
                torch.tensor(labels[1]).float().cuda()[None, None],\
                 agg_flows[-1][pair_id].unsqueeze(0), \
                    mode='nearest')[0,0]
            labels[2] = labels[2].long().cpu().numpy()
            pairs_draw = [label2rgb(labels[i], pairs_img[i].cpu().numpy(), bg_label=0) for i in range(3)]
            pairs_draw_organ = [draw_seg_on_vol(pairs_img[i], pairs_seg_organ[i]) for i in range(3)]
            save_dir = pa('./images/tmp/{}/'.format(args.checkpoint.split('/')[-2]))
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            visualize_3d(pairs_draw_organ[0], save_name=save_dir.parent/'{}_so.png'.format(pairs[pair_id][0]), print_=True, color_channel=1)
            visualize_3d(pairs_draw_organ[1], save_name=save_dir.parent/'{}_so.png'.format(pairs[pair_id][1]), color_channel=1)
            visualize_3d(pairs_draw_organ[2], save_name=save_dir / '{}_{}_warped_so.png'.format(*pairs[pair_id]), color_channel=1)
            visualize_3d(pairs_draw[0], save_name=save_dir.parent / '{}_s.png'.format(pairs[pair_id][0]), print_=True, color_channel=3)
            visualize_3d(pairs_draw[1], save_name=save_dir.parent / '{}_s.png'.format(pairs[pair_id][1]), color_channel=3)
            visualize_3d(pairs_draw[2], save_name=save_dir / '{}_{}_warped_s.png'.format(*pairs[pair_id]), color_channel=3)
            visualize_3d(pairs_img[0], save_name=save_dir.parent / '{}.png'.format(pairs[pair_id][0]), print_=True)
            visualize_3d(pairs_img[1], save_name=save_dir.parent / '{}.png'.format(pairs[pair_id][1]))
            visualize_3d(pairs_img[2], save_name=save_dir / '{}_{}_warped.png'.format(*pairs[pair_id]))
            print('save to {}'.format(save_dir))
            if args.only_vis_target: quit()
        elif args.only_vis_target:
            continue
        # visualize figures
        if True:
            dir = '/home/hynx/regis/recursive-cascaded-networks/figures/fig1'
            # print(args.checkpoint)
            model_name = args.checkpoint.split('/')[-1].split('_')[3]
            dir = os.path.join(dir, model_name)
            # mkdir
            pa(dir).mkdir(parents=True, exist_ok=True)
            # save images
            jacs = jacobian_det(agg_flows[-1], return_det=True)[:,None]
            jacs = F.interpolate(jacs, size=fixed.shape[-3:], mode='trilinear', align_corners=True)
            for i in range(len(fixed)):
                if 'test_3-0' not in id1[i]: continue

                f, m, w = fixed[i,0], moving[i,0], warped[-1][i,0]
                seg_f, seg_m, seg_w = seg1[i,0], seg2[i,0], w_seg2[i,0]
                id = '{}_{}'.format(id1[i], id2[i])
                # draw_seg tumor
                draw = lambda x, y: draw_seg_on_vol(x, y>1.5, inter_dst=5, alpha=0.1)
                combo_imgs(draw(f, seg_f), draw(m, seg_m), draw(w, seg_w), idst=1).save('{}/{}_seg_draw.png'.format(dir, id))
                # combo_imgs(f, m, w).save('{}/{}.png'.format(dir, id))
                # combo_imgs(seg_f, seg_m, seg_w).save('{}/{}_seg.png'.format(dir, id))
                # print('save to {}/{}.png'.format(dir, id))

                im_dct = {'f':f[::5,None, ], 'm':m[::5,None, ], ',w':w[::5,None, ],\
                          'seg_f':seg_f[::5,None, ], 'seg_m':seg_m[::5,None, ], 'seg_w':seg_w[::5,None, ],}

                # draw line of organ, and overlay tumor
                for im, seg in [(w, seg_w)]:
                    # bnd = find_surf(seg_f>0.5, 3, thres=0.8)
                    bnd = find_boundaries(seg.cpu().numpy()>0.5, mode='outer', connectivity=1)
                    print(bnd.max())
                    bnd = torch.from_numpy(bnd).float()

                    gt_bnd = find_boundaries(seg_f.cpu().numpy()>0.5, mode='outer', connectivity=1)
                    gt_bnd = torch.from_numpy(gt_bnd).float().cpu()
                    tum = seg_w>1.5
                    tum = tum.cpu()

                    lb = torch.stack([bnd, gt_bnd])
                    bnd_img = draw_seg_on_vol(im, lb, inter_dst=5, alpha=1)*255
                    bnd_img = bnd_img.to(dtype=torch.uint8)
                    for d in range(len(bnd_img)):
                        bnd_img[d] = draw_segmentation_masks(bnd_img[d], tum[::5][d], colors='yellow', alpha=0.5)

                    bnd_img = bnd_img/255
                    show_img(bnd_img, inter_dst=1).save('{}/{}_bnd.png'.format(dir, id))
                    print('save to {}/{}_bnd.png'.format(dir, id))
                    im_dct['bnd_img'] = bnd_img
                    # import ipdb; ipdb.set_trace()

                ### set jacobian
                if True:
                    sigm_trsf_f = lambda x: torch.sigmoid((x-1.5)*5)
                    jac = jacs[i,0].cpu()
                    jac = sigm_trsf_f(jac)

                    gt_bnd = find_boundaries(seg_w.cpu().numpy()>0.5, mode='outer', connectivity=1)
                    gt_bnd = torch.from_numpy(gt_bnd)
                    gt_tum_bnd = find_boundaries(seg_w.cpu().numpy()>1.5, mode='outer', connectivity=1)
                    gt_tum_bnd = torch.from_numpy(gt_tum_bnd)
                    
                    seg_on_jac = draw_seg_on_vol(jac, torch.stack([gt_bnd, gt_tum_bnd])
                                                 , inter_dst=5, alpha=1)
                    show_img(seg_on_jac, inter_dst=1).save('{}/{}_jac.png'.format(dir, id))
                    # show_img(jac).save('{}/{}_jac.png'.format(dir, id))
                    print('save to {}/{}_jac.png'.format(dir, id))
                    # import ipdb; ipdb.set_trace()



                # show deform field
                if True:          
                    # (pa(dir)/'{}_w'.format(id)).mkdir(parents=True, exist_ok=True)
                    # (pa(dir).parent/'h5'/'{}'.format(id)).mkdir(parents=True, exist_ok=True)
                    p2d = lambda x: T.ToPILImage()(x.cpu())
                    # if 'test_3-0_test_3-1' not in id: continue
                    flow_imgs = []
                    for ii in range(len(f)):
                        # get current flow
                        flow = agg_flows[-1][i][:,ii].permute(1,2,0)
                        # with OnPlt(figsize=(10,10)):
                        #     ax = plt.gca()
                        #     plot_grid(ax, flow.cpu().numpy(), factor=1)
                        #     plt.savefig(pa(dir)/'{}_w/{}_flow.png'.format(id, ii))
                        # normal flow
                        flow = flow/flow.max()
                        flow_img = flow_to_image(flow.cpu().numpy())
                        flow_img = torch.from_numpy(flow_img).permute(2,0,1)
                        # flow_img = T.ToPILImage()()
                        # flow = flow.permute(2,0,1)
                        # flow = hsv2rgb(flow)
                        # flow_img = T.ToPILImage()(flow.cpu())
                        # flow_img.save(pa(dir)/'{}_w/{}_flowrgb3d.png'.format(id, ii))
                        # print('save to {}/{}_w/{}_flowrgb3d.png'.format(dir, id, ii))
                        flow_imgs.append(flow_img)
                    flow_imgs = torch.stack(flow_imgs)/255
                    show_img(flow_imgs, inter_dst=5).save('{}/{}_flow.png'.format(dir, id))
                    print('save to {}/{}_flow.png'.format(dir, id))
                    im_dct['flow_imgs'] = flow_imgs[::5]

                idx = 13
                selected_idx = 5*(idx-1)
                se_idxs = list(range(0,128,5))
                if True:
                    se_dir = '{}/{}/selected'.format(dir,id)
                    pa(se_dir).mkdir(parents=True, exist_ok=True)
                    for k, ims in im_dct.items():
                        for im in range(len(ims)):
                            idx = se_idxs[im]
                            print(ims[im].shape)
                            T.ToPILImage()(ims[im]).save('{}/{}_{}.png'.format(se_dir, k, idx))
                            print('save to {}/{}_{}.png'.format(se_dir, k, idx))
                import ipdb; ipdb.set_trace()
        dices = []

        t_begin_eval = time.time()
        for k,v in segmentation_class_value.items():
            ### specially for mrbrainS dataset
            # if v > 0 and v not in data['segmentation1'].unique():
            #     # print('no {} in segmentation1'.format(k))
            #     continue
            if args.region_dice and v<=2:
                sseg2 = (data['segmentation2'].cuda() > v-0.5) & (data['segmentation2'].cuda() <= 2.5)
                sseg1 = (data['segmentation1'].cuda() > v-0.5) & (data['segmentation1'].cuda() <= 2.5)
                w_sseg2 =  w_seg2 > (v-0.5)
            else:
                sseg1 = data['segmentation1'].cuda() == v
                sseg2 = data['segmentation2'].cuda() == v
                w_sseg2 = (w_seg2>v-0.5) & (w_seg2<v+0.5)
            dice, jac = dice_jaccard(sseg1, w_sseg2)
            key = 'dice_{}'.format(k)
            if key not in results:
                results[key] = []
                metric_keys.append(key)
            results[key].extend(dice.cpu().numpy())
            dices.append(dice.cpu().numpy())
            # add original dice
            if False:
                original_dice, _ = dice_jaccard(sseg1, sseg2)
                key = 'o_dice_{}'.format(k)
                if key not in results:
                    results[key] = []
                    metric_keys.append(key)
                results[key].extend(original_dice.cpu().numpy())
            # calculate size ratio
            if True:
                original_size = torch.sum(sseg2, dim=(1,2,3,4)).float()
                current_size = torch.sum(w_sseg2, dim=(1,2,3,4)).float()
                size_ratio = current_size / original_size
                key = '{}_ratio'.format(k)
                if key not in results:
                    results[key] = []
                    metric_keys.append(key)
                results[key].extend(size_ratio.cpu().numpy())

            ### Calculate surface deviation metrics (surface_dice, hd-95)
            if args.surf_dist:
                key = 'hd95_{}'.format(k)
                surface_distance = [compute_surface_distances(sseg1.cpu().numpy()[i,0], w_sseg2.cpu().numpy()[i,0]) for i in range(seg1.shape[0])]

                hd95 = [compute_robust_hausdorff(s, 95) for s in surface_distance]
                if key not in results:
                    results[key] = []
                    metric_keys.append(key)
                results[key].extend(hd95)
                # surface dice
                key = 'sdice_{}'.format(k)
                surface_dice = [compute_surface_dice_at_tolerance(s, 5) for s in surface_distance]
                if key not in results:
                    results[key] = []
                    metric_keys.append(key)
                results[key].extend(surface_dice)
                # average surface distance
                key = 'asd_{}'.format(k)
                asd = [compute_average_surface_distance(s)[1] for s in surface_distance]
                if key not in results:
                    results[key] = []
                    metric_keys.append(key)
                results[key].extend(asd)
        
        results['id1'].extend(id1)
        results['id2'].extend(id2)
        # add tumor:liver ratio
        tl1_ratio = (seg1>1.5).sum(dim=(1,2,3,4)).float() / (seg1>0.5).sum(dim=(1,2,3,4)).float()
        tl2_ratio = (seg2>1.5).sum(dim=(1,2,3,4)).float() / (seg2>0.5).sum(dim=(1,2,3,4)).float()
        if 'tl1_ratio' not in metric_keys:
            metric_keys.append('tl1_ratio')
            results['tl1_ratio'] = []
        if 'tl2_ratio' not in metric_keys:
            metric_keys.append('tl2_ratio')
            results['tl2_ratio'] = []
        if 'l1l2_ratio' not in metric_keys:
            metric_keys.append('l1l2_ratio')
            results['l1l2_ratio'] = []
        results['tl1_ratio'].extend(tl1_ratio.cpu().numpy())
        results['tl2_ratio'].extend(tl2_ratio.cpu().numpy())
        results['l1l2_ratio'].extend((seg1>.5).sum(dim=(1,2,3,4)).float() / (seg2>.5).sum(dim=(1,2,3,4)).float().cpu().numpy())
        t_end_eval = time.time()

        # print('infer time: {:.2f}s, eval time: {:.2f}s'.format(t_begin_eval-t_infer, t_end_eval-t_begin_eval))
        
        key = 'to_ratio'
        k2 = '{}_ratio'.format(cfg_training.data_type)
        if 'tumor_ratio'  in results and k2 in results:
            if key not in results:
                results[key] = []
                metric_keys.append(key)
            tumor_ratio = np.array(results['tumor_ratio'][-len(seg1):])
            organ_ratio = np.array(results[k2][-len(seg1):])
            to_ratio = tumor_ratio / organ_ratio
            strr = np.where(to_ratio<1, 1/to_ratio, to_ratio)**2
            results[key].extend(strr)
            for k in range(len(to_ratio)):
                print('to_ratio for {} to {}: {:.4f}'.format(id1[k], id2[k], to_ratio[k]**2))
               
        del fixed, moving, warped, flows, agg_flows, affine_params
        # get mean of dice class
        results['dices'].extend(np.mean(dices, axis=0)) 

    # save result
    with open(output_fname, 'w') as fo:
        # list result keys (each one takes space 8)
        print('{:<30}'.format('id1'), '{:<30}'.format('id2'), '{:<12}'.format('avg_dice'), *['{:<12}'.format(k) for k in metric_keys], file=fo)
        for i in range(len(results['dices'])):
            print('{:<30}'.format(results['id1'][i]), '{:<30}'.format(results['id2'][i]), '{:<12.4f}'.format(np.mean(results['dices'][i])), 
                *['{:<12.4f}'.format(results[k][i]) for k in metric_keys], file=fo)
            # print(results['id1'][i], results['id2'][i], np.mean(results['dices'][i]), *[results[k][i] for k in metric_keys], file=fo)
        # write summary
        print('Summary', file=fo)
        dices = results['dices']
        print("Dice score: {} ({})".format(np.mean(dices), np.std(
            np.mean(dices, axis=-1))), file=fo)
        # Dice score for organ and tumour
        # sort metric_keys
        metric_keys = sorted(metric_keys)
        for k in metric_keys:
            # nan exclude
            print("{}: {} ({})".format(k, np.nanmean(results[k]), np.nanstd(results[k])), file=fo)

    print('Saving to {}'.format(output_fname))
    # create dir
    os.makedirs(os.path.dirname(output_fname), exist_ok=True)
    if args.save_pkl:
        pkl_name = output_fname.replace('.txt', '.pkl')
        with open(pkl_name, 'wb') as fo:
            pickle.dump(results, fo)
        print('finish saving pkl to {}'.format(pkl_name))

if __name__ == '__main__':
    main()

