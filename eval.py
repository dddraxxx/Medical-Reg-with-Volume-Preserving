import argparse
from genericpath import isfile
import os
import pickle
import json
import re
import numpy as np
from metrics.losses import score_metrics
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch

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
args = parser.parse_args()

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
    model = RecursiveCascadeNetwork(n_cascades=args.n_cascades, im_size=image_size).cuda()
    # add checkpoint loading
    print("Loading checkpoint from {}".format(args.checkpoint))
    if os.path.isdir(args.checkpoint):
        load_model_from_dir(args.checkpoint, model)
    else:
        model_path = args.checkpoint
        load_model(torch.load(model_path), model)
    
    # parent of model path
    output_fname = './evaluations/{}_{}_{}.txt'.format(os.path.dirname(model_path).split('/')[-1], args.val_subset or Split.VALID, args.name or '')
    print('Saving to {}'.format(output_fname))
    # create dir
    os.makedirs(os.path.dirname(output_fname), exist_ok=True)

    # run val
    model.eval()
    results = {}
    results['id1'], results['id2'], results['dices'] = [], [], []
    if args.save_pkl:
        results['agg_flows'] = []
        results['affine_params'] = []
    for iteration, data in tqdm(enumerate(val_loader)):
        fixed, moving = data['voxel1'], data['voxel2']
        id1, id2 = data['id1'], data['id2']
        fixed = fixed.cuda()
        moving = moving.cuda()
        warped, flows, agg_flows, affine_params = model(fixed, moving, return_affine=True)
        
        if args.save_pkl:
            magg_flows = torch.stack(agg_flows).transpose(0,1).detach().cpu()
            results['agg_flows'].extend(magg_flows)
            results['affine_params'].extend(affine_params['theta'].detach().cpu())
        # metrics: dice
        results['id1'].extend(id1)
        results['id2'].extend(id2)
        dices = []
        for k,v in segmentation_class_value.items():
            seg1 = data['segmentation1'].cuda() > v-0.5
            seg2 = data['segmentation2'].cuda() > v-0.5
            w_seg2 = model.reconstruction(seg2.float(), agg_flows[-1].float()) > 0.5
            dice, jac = score_metrics(seg1, w_seg2)
            key = 'dice_{}'.format(k)
            if key not in results:
                results[key] = []
            results[key].extend(dice.cpu().numpy())
            dices.append(dice.cpu().numpy())
        # get mean of dice class
        results['dices'].extend(np.mean(dices, axis=0)) 

    # save result
    with open(output_fname, 'w') as fo:
        keys = ['dice_{}'.format(k) for k in segmentation_class_value]
        for i in range(len(results['dices'])):
            print(results['id1'][i], results['id2'][i], np.mean(results['dices'][i]), *[results[k][i] for k in keys], file=fo)
        print('Summary', file=fo)
        dices = results['dices']
        print("Dice score: {} ({})".format(np.mean(dices), np.std(
            np.mean(dices, axis=-1))), file=fo)
        # Dice score for organ and tumour
        for dice_k in keys:
            print("{}: {} ({})".format(dice_k, np.mean(results[dice_k]), np.std(results[dice_k])), file=fo)

    if args.save_pkl:
        with open(output_fname.replace('.txt', '.pkl'), 'wb') as fo:
            pickle.dump(results, fo)
        print('finish saving pkl')

if __name__ == '__main__':
    main()

