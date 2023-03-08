# %%
import ml_collections
from timeit import default_timer
import json
import argparse
import os
import time

import numpy as np
from run_utils import build_precompute
from tools.utils import *
import torch
from torch.functional import F
from networks.recursive_cascade_networks import RecursiveCascadeNetwork
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.data import DataLoader
from metrics.losses import det_loss, focal_loss, jacobian_det, masked_sim_loss, ortho_loss, reg_loss, dice_jaccard, sim_loss, similarity, surf_loss, dice_loss
import datetime as datetime
from torch.utils.tensorboard import SummaryWriter
# from data_util.ctscan import sample_generator
from data_util.dataset import Data, Split
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
sys.argv = ['']

# default conifg
# -g 3 --name stage1 -m hard -vp 0.1 -st organ --debug -ts mini
args=['-g', '3', '--name', 'stage1', '-m', 'soft', '-vp', '0.1', '-st', 'organ', '--debug', '-ts', 'lits', '-base', 'VTN', '-ts', 'lits', '-use2', '1', '-useb','1']
from train import parser
args = parser.parse_args(args)
import subprocess
GPU_ID = subprocess.getoutput('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs')
print('Using GPU', GPU_ID)
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

# %%
if 'LOCAL_RANK' in os.environ:
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
else:
    local_rank = 0
# Hong Kong time
dt = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
run_id = dt.strftime('%b%d_%H%M%S') + (args.name and '_' + args.name)

ckp_freq = int(args.checkpoint_frequency * args.round)
data_type = 'liver' if 'liver' in args.dataset else 'brain'
args.data_type = data_type

# read config
with open(args.dataset, 'r') as f:
    cfg = json.load(f)
    image_size = cfg.get('image_size', [128, 128, 128])
    image_type = cfg.get('image_type', None)
    segmentation_class_value=cfg.get('segmentation_class_value', {'unknown':1})

# %%
train_scheme = args.training_scheme or Split.TRAIN
dataset = Data(args.dataset, rounds=None, scheme=train_scheme)

# %%
in_channels = args.in_channel
model = RecursiveCascadeNetwork(n_cascades=args.n_cascades, im_size=image_size, base_network=args.base_network, in_channels=in_channels).cuda()
if args.masked in ['soft', 'hard']:
    cfg_train = ml_collections.ConfigDict(args.__dict__)
    precompute_h5 = False
    s1_model = build_precompute(model, dataset, cfg_train).RCN

    
trainable_params = []
for submodel in model.stems:
    trainable_params += list(submodel.parameters())
trainable_params += list(model.reconstruction.parameters())

start_epoch = 0
start_iter = 0

num_worker = min(8, args.batch_size)
loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_worker, shuffle=True)

# use cudnn.benchmark
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# %%
# write_h5 = '/home/hynx/regis/recursive-cascaded-networks/datasets/{}-{}_computed.h5'.format(args.training_scheme, args.base_network)
# # r+ for h5py
# import h5py
# if os.path.exists(write_h5):
#     print('h5 file exists, append', write_h5)
#     h5f = h5py.File(write_h5, 'r+')
# else:
#     print('write h5 file', write_h5)
#     h5f = h5py.File(write_h5, 'w')

# %%
# id2s = set([h for h in h5f.keys() if args.training_scheme in h])
id2s = set()
l = len(dataset.subset[dataset.scheme])
print(len(id2s), l)
fixed_fixed = model.pre_register.template_input
fixed_fixed.shape # torch.Size([1, 1, 128, 128, 128])

## Try to precompute mask
import tqdm
tq = tqdm.trange(len(id2s))
len(id2s), l
results = {
    'dice_soft': [],
    'dice_hard': [],
    'dice_organ': [],
    'dice_unsup': [],
    'tumor_rat_gt': [],
    'tumor_rat_input': [],
}
#%%
# create tqdm object without knowing the limit
tq = tqdm.tqdm()
for iteration, data in enumerate(loader):
    # if len(id2s)>=l:
    #     break
    # torch clear cache
    # torch.cuda.empty_cache()
    # if start_iter is not 0, the 'if' statement will work
    # if iteration>=len(loader): break
    # id2 = data['id2']
    # selected = np.array([i for i in range(len(id2)) if id2[i] not in id2s])
    # moving, fixed, seg1, seg2 = moving[selected], fixed[selected], seg1[selected], seg2[selected]
    # id1, id2 = np.array(id1), np.array(id2)
    # id1, id2 = id1[selected], id2[selected]
    
    # if loop > 1000, stop
    if iteration>500/4: break

    seg2 = data['segmentation2'].cuda()
    selected = (seg2>1.5).sum(dim=(1,2,3,4))>0
    selected = selected.cpu().numpy()
    if len(selected)==0: continue
    for k in data.keys():
        if k in ['id1', 'id2']:
            data[k] = np.array(data[k])[selected]
        else: data[k] = data[k][selected]

    fixed, moving = data['voxel1'], data['voxel2']
    fixed = fixed.cuda()
    moving = moving.cuda()
    seg1 = data['segmentation1'].cuda()
    seg2 = data['segmentation2'].cuda()
    id1 = data['id1']


    ### Calc similarity to decide whether to precompute?
    if False:
        with torch.no_grad():
            warped, flows, agg_flows = s1_model(fixed, moving, return_affine=False)
            sim = similarity(fixed, warped[-1])
            selected = (sim>0.83)

        id2 = np.array(id2)[selected]
        id1 = np.array(id1)[selected]
        selected=  selected.squeeze().cpu().nonzero().squeeze(1).numpy()
        # print(selected, id2)

        id2 = id2[selected]
        sim = sim.squeeze().cpu().numpy()[selected]
        # print(id2)
        h5_keys = list(h5f.keys())
        if len(selected)==0: continue

        id1 = id1[selected]
        id2 = id2[selected]
        fixed = fixed[selected]
        moving = moving[selected]
        print(selected, id2, sim[selected])
    ### End

    # computed mask: generate mask for tumor region according to flow
    args.mask_threshold = 1.5
    cfg_train = ml_collections.ConfigDict(args.__dict__)
    cfg_train['masked'] = 'soft'
    cfg_train['use_2nd_flow'] = True
    # cfg_train['masked_threshold'] = 2
    fixed = fixed_fixed.expand_as(fixed)
    input_seg, compute_mask = model.pre_register(fixed, moving, None, training=False, cfg=cfg_train)

    # id2s = id2s.union(set(id2))
    # tq.write(str(len(id2s)))
    tq.update(len(compute_mask))
    # print(cfg_train)
    # print(compute_mask.unique())
    # add dice to result
    if True:
        with torch.no_grad():
            # only for tumor
            # dice, jac = dice_jaccard(compute_mask, seg2>1.5)
            dice, jac = dice_jaccard(compute_mask>0.2, seg2>1.5)
            results['dice_hard'].extend(dice)
            dice, jac = dice_jaccard(compute_mask, seg2>1.5)
            results['dice_soft'].extend(dice)
            dice, jac = dice_jaccard(input_seg>0.5, seg2>0.5)
            results['dice_organ'].extend(dice)
            dice, jac = dice_jaccard(input_seg>0.5, seg2>1.5)
            results['dice_unsup'].extend(dice)
            tumor_rat_gt = ((compute_mask>0.5)& (seg2>1.5)).sum(dim=(1,2,3,4)).float() / (seg2>1.5).sum(dim=(1,2,3,4)).float()
            tumor_rat_input = ((compute_mask>0.2)& (seg2>0.5)).sum(dim=(1,2,3,4)).float() / (compute_mask>0.2).sum(dim=(1,2,3,4)).float()
            results['tumor_rat_gt'].extend(tumor_rat_gt)
            results['tumor_rat_input'].extend(tumor_rat_input)
        for k in results:
            print(k, results[k][-1])

    if False:
        # tq.update(len(id2))
        for i in range(len(id2)):
            h5f.create_group(id2[i])
            h5f[id2[i]].create_dataset('input_seg', data=input_seg[i].cpu().numpy(), dtype='float32')
            h5f[id2[i]].create_dataset('compute_mask', data=compute_mask[i].cpu().numpy(), dtype='float32')

#%% rm 0s in results
for k in results:
    results[k] = [x for x in results[k] if x>0.01]
#%%
for k in results:
    print(len(results[k]))
    print(k, torch.stack(results[k]).mean())
# %%
# h5f.close()
print(len(id2s))


