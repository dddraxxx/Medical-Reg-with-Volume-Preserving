import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

from _ants import ants_pred
import torchvision.transforms as T
from metrics.losses import *
import time
from pathlib import Path as pa
import argparse
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
parser.add_argument('-rd', '--region_dice', default=True, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='If calculate dice for each region')
parser.add_argument('-sd', '--surf_dist', default=False, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='If calculate dist for each surface')
parser.add_argument('-ua','--use_ants', action='store_true', help='if use ants to register')
parser.add_argument('--debug', action='store_true', help='if debug')
args = parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
else:
    # set gpu to the one with most free memory
    import subprocess
    GPU_ID = subprocess.getoutput('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs')
    print('Using GPU', GPU_ID)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

# resolve checkpoint path to realpath
args.checkpoint = pa(args.checkpoint).resolve().__str__()

def main(args):
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
    cfg_training.hyper_vp = hasattr(cfg_training, "hyper_vp") and cfg_training.hyper_vp
    model = RecursiveCascadeNetwork(n_cascades=cfg_training.n_cascades, im_size=image_size, base_network=cfg_training.base_network, in_channels=2+bool(cfg_training.masked), hyper_net=cfg_training.hyper_vp).cuda()
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
    output_fname = './eval/evaluations/{}_{}.txt'.format(args.val_subset or Split.VALID, exp_name)
    output_fname = os.path.abspath(output_fname)
    print('will save to', output_fname)
    if not os.path.exists(output_fname):
        os.makedirs(os.path.dirname(output_fname), exist_ok=True)

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
        results['flow'] = []
        results['affine_params'] = []
        if args.reverse:
            results['rev_flow'] = []
        results['warped'] = []
        results.update({
            "img1": [],
            "img2": [],
            "seg1": [],
            "seg2": [],
            "wseg2": [],
        })
    metric_keys = []

    def pick_data(idx, data):
        '''idx: batch-dim mask'''
        for k in data.keys():
            if 'id' in k:
                data[k] = [i for l,i in zip(idx, data[k]) if l]
            else: data[k] = data[k][idx,...]
        return data


    for iteration, data in tqdm(enumerate(val_loader)):
        seg1, seg2 = data['segmentation1'], data['segmentation2']
        seg1, seg2 = data['segmentation1'].float(), data['segmentation2'].float()

        fixed, moving = data['voxel1'], data['voxel2']
        id1, id2 = data['id1'], data['id2']
        if args.use_ants:
            pred = ants_pred(fixed, moving, seg2)
            w_seg2, warped = pred['w_seg2'], pred['warped']
            w_seg2 = torch.from_numpy(w_seg2).float().cuda()
            warped = torch.from_numpy(warped).float().cuda()
        else:
            with torch.no_grad():
                fixed = fixed.cuda()
                moving = moving.cuda()
                seg2 = seg2.cuda()
                if cfg_training.masked  in ['soft' , 'hard']:
                    input_seg, compute_mask = model.pre_register(fixed, moving, seg2, training=False, cfg=cfg_training)
                    moving_ = torch.cat([moving, input_seg.float().cuda()], dim=1)
                else:
                    moving_ = moving
                warped_, flows, agg_flows, affine_params = model(fixed, moving_, return_affine=True,)

            warped = [model.reconstruction(moving, agg_flows[-1].float())]
            w_seg2 = model.reconstruction(seg2.float(), agg_flows[-1].float())

        if args.save_pkl:
            # now we just save the last flow
            magg_flow = agg_flows[-1].detach().cpu()

            results['flow'].extend(magg_flow)
            results['affine_params'].extend(affine_params['theta'].detach().cpu())
            results['warped'].extend(warped[-1])
            results['img1'].extend(fixed.detach().cpu())
            results['img2'].extend(moving.detach().cpu())
            results['wseg2'].extend(w_seg2.detach().cpu())
            results['seg1'].extend(seg1.detach().cpu())
            results['seg2'].extend(seg2.detach().cpu())
            # results["id1"].extend(id1)
            # results["id2"].extend(id2)

        dices = []

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
            if False:
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

        # add J>0 ratio, and std |J|
        if not args.use_ants:
            flow = agg_flows[-1]
            jacs = jacobian_det(flow, return_det=True)
            if 'std_jac' not in results:
                results['std_jac'] = []
                metric_keys.append('std_jac')
            if 'Jleq0' not in results:
                results['Jleq0'] = []
                metric_keys.append('Jleq0')
            results['std_jac'].extend(jacs.flatten(1).std(1).cpu().numpy())
            results['Jleq0'].extend((jacs.flatten(1)<=0).sum(1).cpu().numpy()/jacs.flatten(1).shape[1])
        # import ipdb; ipdb.set_trace()

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
        k2 = '{}_ratio'.format(cfg_training.get('data_type', 'brain' if 'brain' in args.checkpoint else 'liver'))
        if 'tumor_ratio'  in results and k2 in results:
            if key not in results:
                results[key] = []
                metric_keys.append(key)
            tumor_ratio = np.array(results['tumor_ratio'][-len(seg1):])
            organ_ratio = np.array(results[k2][-len(seg1):])
            to_ratio = tumor_ratio / organ_ratio
            strr = np.where(to_ratio<1, 1/to_ratio, to_ratio)**2
            results[key].extend(strr)
            # for k in range(len(to_ratio)):
            #     print('to_ratio for {} to {}: {:.4f}'.format(id1[k], id2[k], to_ratio[k]**2))

        if not args.use_ants:
            del fixed, moving, warped, flows, agg_flows, affine_params
        # get mean of dice class
        results['dices'].extend(np.mean(dices, axis=0))

    # save result
    with open(output_fname, 'w') as fo:
        # list result keys (each one takes space 8)
        print('{:<30}'.format('id1'), '{:<30}'.format('id2'), '{:<12}'.format('avg_dice'), *['{:<12}'.format(k) for k in metric_keys], file=fo)
        # import ipdb; ipdb.set_trace()
        # print([(k, len(results[k])) for k in results.keys()])
        for i in range(len(results['dices'])):
            # print(k)
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
        # get dir name
        pkl_name = output_fname.replace('.txt', '.pkl').replace('evaluations','eval_pkls')
        os.makedirs(os.path.dirname(pkl_name), exist_ok=True)
        with open(pkl_name, 'wb') as fo:
            pickle.dump(results, fo)
        print('finish saving pkl to {}'.format(pkl_name))

if __name__ == '__main__':
    main(args)
