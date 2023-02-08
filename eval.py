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

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, default=None,
                    help='Specifies a previous checkpoint to load')
parser.add_argument('-n', "--n_cascades", type=int, default=3)
parser.add_argument('-r', '--rep', type=int, default=1,
                    help='Number of times of shared-weight cascading')
parser.add_argument('-g', '--gpu', type=str, default='0',
                    help='Specifies gpu device(s)')
parser.add_argument('-d', '--dataset', type=str, default=None,
                    help='Specifies a data config')
parser.add_argument('-v', '--val_subset', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=4, help='Size of minibatch')
parser.add_argument('--data_args', type=str, default=None)
parser.add_argument('--net_args', type=str, default=None)
parser.add_argument('--name', type=str, default=None)
parser.add_argument('-s','--save_pkl', action='store_true', help='Save the results as a pkl file')
parser.add_argument('-base', '--base_network', type=str, default='VTN')
parser.add_argument('-re','--reverse', action='store_true', help='If save reverse flow in pkl file')
parser.add_argument('-tl','--test_large', action='store_true', help='If test on data with small tumor')
parser.add_argument('-tb','--test_boundary', action='store_true', help='If test on data with tumor close to organ boundary')
parser.add_argument('-lm', '--lmd', action='store_true', help='If test landmark locations')
parser.add_argument('--lmk_json', type=str, default='/home/hynx/regis/recursive-cascaded-networks/landmark_json/lits17_landmark.json', help='landmark for eval files')
parser.add_argument('-m', '--masked', action='store_true', help='If model need masks')
parser.add_argument('-lm_r', '--lmk_radius', type=int, default=10, help='affected landmark within radius')
parser.add_argument('-vl', '--visual_lmk', action='store_true', help='If visualize landmark')
parser.add_argument('-rd', '--region_dice', default=True, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='If calculate dice for each region')
parser.add_argument('-sd', '--surf_dist', default=True, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='If calculate dist for each surface')
parser.add_argument('-mt', '--masked_type', type=str, default='seg', help='masked type')
parser.add_argument('-only_vis', '--only_vis_target', action='store_true', help='If only visualize target')
args = parser.parse_args()
if args.checkpoint == 'normal':
    args.checkpoint = '/home/hynx/regis/recursive-cascaded-networks/logs/Dec27_133859_normal/model_wts'
    # args.checkpoint = '/home/hynx/regis/recursive-cascaded-networks/logs/Dec06_012136_normal-vtn/model_wts'

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

def main():
    # build dataset
    # read config
    with open(args.dataset, 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        image_type = cfg.get('image_type', None)
        segmentation_class_value=cfg.get('segmentation_class_value', {'unknown':1})
    val_dataset = Data(args.dataset, scheme=args.val_subset or Split.VALID)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=min(8, args.batch_size), shuffle=False)
    # build framework
    model = RecursiveCascadeNetwork(n_cascades=args.n_cascades, im_size=image_size, base_network=args.base_network, in_channels=2+args.masked).cuda()
    # add checkpoint loading
    print("Loading checkpoint from {}".format(args.checkpoint))
    from tools.utils import load_model, load_model_from_dir
    model_path = args.checkpoint
    if os.path.isdir(args.checkpoint):
        model_path = load_model_from_dir(args.checkpoint, model)
    else:
        load_model(torch.load(model_path), model)
    
    # parent of model path
    import re
    # "([^\/]*_\d{6}_[^\/]*)"gm
    exp_name = re.search(r"([^\/]*_\d{6}_[^\/]*)", model_path).group(1)
    output_fname = './evaluations/{}_{}{}_{}.txt'.format(exp_name, (args.name and f'{args.name}_') or '', args.val_subset or Split.VALID, '' if not args.lmd else 'lm{}'.format(args.lmk_radius))
    print('will save to', output_fname)

    # stage 1 model setup
    if args.masked_type in ['soft', 'hard']:
        template = list(val_dataset.subset['slits-temp'].values())[0]
        template_image, template_seg = template['volume'], template['segmentation']
        template_image = torch.tensor(np.array(template_image).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()/255.0
        template_seg = torch.tensor(np.array(template_seg).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        print('Using pre-register model as stage1')
        model.build_preregister(template_image, template_seg).RCN.cuda()
        model.compute_mask=True

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
        seg1, seg2 = data['segmentation1'], data['segmentation2']
        # seg1, seg2 = (seg1>0.5).float(), (seg2>0.5).float()
                
        fixed, moving = data['voxel1'], data['voxel2']
        id1, id2 = data['id1'], data['id2']
        with torch.no_grad():
            fixed = fixed.cuda()
            moving = moving.cuda()
            if args.masked and args.masked_type =='seg':
                moving_ = torch.cat([moving, seg2.float().cuda()], dim=1)
            else:
                moving_ = moving
            warped_, flows, agg_flows, affine_params = model(fixed, moving_, return_affine=True, return_neg=args.reverse)
            warped = [i[:,:1,...] for i in warped_]
            if args.masked:
                w_seg2 = warped_[-1][:, 1:]
            else:
                w_seg2 = model.reconstruction(seg2.float().cuda(), agg_flows[-1].float())
        
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
        # target_pair = ('lits_51', '51')
        # target_pairs = [('lits_{}'.format(51), 'lits_{}'.format(33))]
        target_pair = ('lits_{}'.format(51), 'yanx_{}'.format(14))
        # target_pair = ('lits_{}'.format(115), 'lits_{}'.format(128))
        # if any([p in pairs for p in target_pairs]):
        if target_pair in pairs:
            from tools.utils import visualize_3d, draw_seg_on_vol, show_img
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
        dices = []

        for k,v in segmentation_class_value.items():
            if args.region_dice:
                seg1 = data['segmentation1'].cuda() > v-0.5
                seg2 = data['segmentation2'].cuda() > v-0.5
            else:
                seg1 = data['segmentation1'].cuda() == v
                seg2 = data['segmentation2'].cuda() == v
            w_seg2 = model.reconstruction(seg2.float(), agg_flows[-1].float()) > 0.5
            dice, jac = dice_jaccard(seg1, w_seg2)
            key = 'dice_{}'.format(k)
            if key not in results:
                results[key] = []
                metric_keys.append(key)
            results[key].extend(dice.cpu().numpy())
            dices.append(dice.cpu().numpy())
            # add original dice
            original_dice, _ = dice_jaccard(seg1, seg2)
            key = 'o_dice_{}'.format(k)
            if key not in results:
                results[key] = []
                metric_keys.append(key)
            results[key].extend(original_dice.cpu().numpy())
            # calculate size ratio
            original_size = torch.sum(seg2, dim=(1,2,3,4)).float()
            current_size = torch.sum(w_seg2, dim=(1,2,3,4)).float()
            size_ratio = current_size / original_size
            key = '{}_ratio'.format(k)
            if key not in results:
                results[key] = []
                metric_keys.append(key)
            results[key].extend(size_ratio.cpu().numpy())

            ### Calculate surface deviation metrics (surface_dice, hd-95)
            if args.surf_dist:
                key = 'hd95_{}'.format(k)
                surface_distance = [compute_surface_distances(seg1.cpu().numpy()[i,0], w_seg2.cpu().numpy()[i,0]) for i in range(seg1.shape[0])]

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
        key = 'to_ratio'
        if key not in results:
            results[key] = []
            metric_keys.append(key)
        tumor_ratio = np.array(results['tumor_ratio'][-len(seg1):])
        organ_ratio = np.array(results['liver_ratio'][-len(seg1):])
        to_ratio = tumor_ratio / organ_ratio
        results[key].extend(np.where(to_ratio<1, 1/to_ratio, to_ratio)**2)

                
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

