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

parser = argparse.ArgumentParser()
parser.add_argument('-bs', "--batch_size", type=int, default=4)
parser.add_argument('-base', '--base_network', type=str, default='VTN')
parser.add_argument('-n', "--n_cascades", type=int, default=3)
parser.add_argument('-e', "--epochs", type=int, default=5)
parser.add_argument("-r", "--round", type=int, default=20000)
parser.add_argument("-v", "--val_steps", type=int, default=1000)
parser.add_argument('-cf', "--checkpoint_frequency", default=0.5, type=float)
parser.add_argument('-c', "--checkpoint", type=str, default=None)
parser.add_argument('-ct', '--continue_training', action='store_true')
parser.add_argument('--ctt', '--continue_training_this', type=str, default=None)
parser.add_argument('--fixed_sample', type=int, default=100)
parser.add_argument('-g', '--gpu', type=str, default='', help='GPU to use')
parser.add_argument('-d', '--dataset', type=str, default='datasets/liver_cust.json', help='Specifies a data config')
parser.add_argument("-ts", "--training_scheme", type=lambda x:int(x) if x.isdigit() else x, default='', help='Specifies a training scheme')
parser.add_argument('-vs', '--val_scheme', type=lambda x:int(x) if x.isdigit() else x, default='', help='Specifies a validation scheme')
parser.add_argument('-aug', '--augment', default=True, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='Augment data')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_scheduler', default = 'step', type=str, help='lr scheduler', choices=['linear', 'step', 'cosine'])
parser.add_argument('--debug', action='store_true', help="run the script without saving files")
parser.add_argument('--name', type=str, default='')
parser.add_argument('--ortho', type=float, default=0.1, help="use ortho loss")
parser.add_argument('--det', type=float, default=0.1, help="use det loss")
parser.add_argument('--reg', type=float, default=1, help="use reg loss")
parser.add_argument('-m', '--masked', choices=['soft', 'seg', 'hard'], default='',
     help="mask the tumor part when calculating similarity loss")
parser.add_argument('-mt', '--mask_threshold', type=float, default=2, help="volume changing threshold for mask")
parser.add_argument('-mn', '--masked_neighbor', type=int, default=3, help="for masked neibor calculation")
parser.add_argument('-vp', '--vol_preserve', type=float, default=0, help="use volume-preserving loss")
parser.add_argument('-st', '--size_type', choices=['organ', 'tumor', 'tumor_gt', 'constant', 'dynamic', 'reg'], default='tumor', help = 'organ means VP works on whole organ, tumor means VP works on tumor region, tumor_gt means VP works on tumor region with ground truth, constant means VP ratio is a constant, dynamic means VP has dynamic weight, reg means VP is replaced by reg loss')
parser.add_argument('--ks_norm', default='voxel', choices=['image', 'voxel'])
parser.add_argument('-w_ksv', '--w_ks_voxel', default=1, type=float, help='Weight for voxel method in ks loss')
parser.add_argument('-s1r', '--stage1_rev', type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=False, help="whether to use reverse flow in stage 1")
parser.add_argument('-inv', '--invert_loss', action='store_true', help="invertibility loss")
parser.add_argument('--surf_loss', default=0, type=float, help='Surface loss weight')
parser.add_argument('-dc', '--dice_loss', default=0, type=float, help='Dice loss weight')
parser.add_argument('-ic', '--in_channel', default=2, type=int, help='Input channel number')
parser.add_argument('-trsf', '--soft_transform', default='sigm', type=str, help='Soft transform')
parser.add_argument('-bnd_thick', '--boundary_thickness', default=0, type=float, help='Boundary thickness')
parser.add_argument('-use2', '--use_2nd_flow', default=False, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='Use 2nd flow')
# mask calculation needs an extra mask as input
import sys
sys.argv = ['']
parser.set_defaults(in_channel=3 if parser.parse_args().masked else 2)
parser.set_defaults(stage1_rev=True if parser.parse_args().base_network == 'VXM' else False)

# default conifg
# -g 3 --name stage1 -m hard -vp 0.1 -st organ --debug -ts mini
args=['-g', '3', '--name', 'stage1', '-m', 'soft', '-vp', '0.1', '-st', 'organ', '--debug', '-ts', 'lits', '-base', 'VTN', '-ts', 'lits']
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
#    # %%
# # data.data_pairs = 
# pairs = [data.data_pairs[i*(l-1):i*(l-1)+l-1] for i in range(l)]
# pairs_id = [[(p[0]['id'], p[1]['id']) for p in pair] for pair in pairs]
# pairs_id [-3:]

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
}
for iteration, data in enumerate(loader):
    if len(id2s)>=l:
        break
    # torch clear cache
    # torch.cuda.empty_cache()
    # if start_iter is not 0, the 'if' statement will work
    if iteration>=len(loader): break
    id2 = data['id2']
    selected = np.array([i for i in range(len(id2)) if id2[i] not in id2s])
    if len(selected)==0: continue

    fixed, moving = data['voxel1'], data['voxel2']
    fixed = fixed.cuda()
    moving = moving.cuda()
    seg1 = data['segmentation1'].cuda()
    seg2 = data['segmentation2'].cuda()
    id1 = data['id1']

    moving, fixed, seg1, seg2 = moving[selected], fixed[selected], seg1[selected], seg2[selected]
    id1, id2 = np.array(id1), np.array(id2)
    id1, id2 = id1[selected], id2[selected]

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
    fixed = fixed_fixed.expand_as(fixed)
    input_seg, compute_mask = model.pre_register(fixed, moving, None, training=False, cfg=cfg_train)

    id2s = id2s.union(set(id2))
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
    if False:
        # tq.update(len(id2))
        for i in range(len(id2)):
            h5f.create_group(id2[i])
            h5f[id2[i]].create_dataset('input_seg', data=input_seg[i].cpu().numpy(), dtype='float32')
            h5f[id2[i]].create_dataset('compute_mask', data=compute_mask[i].cpu().numpy(), dtype='float32')

#%%
for k in results:
    print(k, torch.stack(results[k]).mean())
# %%
# h5f.close()
print(len(id2s))


