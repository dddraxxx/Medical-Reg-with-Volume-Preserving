#%%
from timeit import default_timer
import json
import argparse
import os
import time

import numpy as np
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
args=['-g', '3', '--name', 'stage1', '-m', 'hard', '-vp', '0.1', '-st', 'organ', '--debug', '-ts', 'lits', '-base', 'VTN', '-ts', 'sliver']
args = parser.parse_args(args)
import subprocess
GPU_ID = subprocess.getoutput('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs')
print('Using GPU', GPU_ID)
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

def reduce_mean(tensor):
    if dist.is_initialized():
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt
    else:
        return tensor

#%%
if True:
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

    # read config
    with open(args.dataset, 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        image_type = cfg.get('image_type', None)
        segmentation_class_value=cfg.get('segmentation_class_value', {'unknown':1})
    #%%
    train_scheme = args.training_scheme or Split.TRAIN
    val_scheme = args.val_scheme or Split.VALID
    print('Train', train_scheme)
    print('Val', val_scheme)
    data = Data(args.dataset, rounds=None, scheme=train_scheme)
#    # %%
    # # data.data_pairs = 
    l = len(data.subset[data.scheme])
    # pairs = [data.data_pairs[i*(l-1):i*(l-1)+l-1] for i in range(l)]
    # pairs_id = [[(p[0]['id'], p[1]['id']) for p in pair] for pair in pairs]
    # pairs_id [-3:]
    #%%
    in_channels = args.in_channel
    model = RecursiveCascadeNetwork(n_cascades=args.n_cascades, im_size=image_size, base_network=args.base_network, in_channels=in_channels, compute_mask=False).cuda()
    if args.masked in ['soft', 'hard']:
        template = list(data.subset['temp'].values())[0]
        template_image, template_seg = template['volume'], template['segmentation']
        template_image = torch.tensor(np.array(template_image).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()/255.0
        template_seg = torch.tensor(np.array(template_seg).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        if data_type == 'liver':
            if args.base_network == 'VTN':
                state_path = '/home/hynx/regis/recursive-cascaded-networks/logs/Jan08_180325_normal-vtn'
            elif args.base_network == 'VXM':
                # state_path = '/home/hynx/regis/recursive-cascaded-networks/logs/Jan08_180325_normal-vtn'
                state_path = '/home/hynx/regis/recursive-cascaded-networks/logs/Jan08_175614_normal-vxm'
        elif data_type == 'brain':
            if args.base_network == 'VTN':
                state_path = '/home/hynx/regis/recursive-cascaded-networks/logs/brain/Feb27_034554_br-normal'
            elif args.base_network == 'VXM':
                state_path = ''
        if args.stage1_rev:
            print('using rev_flow')
        s1_model = model.build_preregister(template_image, template_seg, state_path, args.base_network).RCN

        
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        mmodel = model.module
    else: mmodel = model
    trainable_params = []
    for submodel in mmodel.stems:
        trainable_params += list(submodel.parameters())
    trainable_params += list(mmodel.reconstruction.parameters())

    start_epoch = 0
    lr = args.lr
    optim = Adam(trainable_params, lr=lr, weight_decay=1e-4)
    if args.lr_scheduler == 'step':
        scheduler = StepLR(optimizer=optim, step_size=10, gamma=0.96)
    elif args.lr_scheduler == 'linear':
        min_lr = 1e-6
        scheduler = LambdaLR(optimizer=optim, lr_lambda=lambda epoch: min_lr + (lr - min_lr) * (1 - epoch / args.epochs), last_epoch= start_epoch-1)

    # add checkpoint loading
    start_epoch = 0
    start_iter = 0

    num_worker = min(8, args.batch_size)
    if dist.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=True)
        loader = DataLoader(data, batch_size=args.batch_size, num_workers=num_worker, sampler=train_sampler)
    else:
        loader = DataLoader(data, batch_size=args.batch_size, num_workers=num_worker, shuffle=True)

    # Saving the losses
    train_loss_log = []
    reg_loss_log = []
    val_loss_log = []
    max_seg = max(segmentation_class_value.values())

    # use cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(3749)
    #%%
    write_h5 = '/home/hynx/regis/recursive-cascaded-networks/datasets/{}_computed.h5'.format(args.base_network)
    # r+ for h5py
    import h5py
    if os.path.exists(write_h5):
        print('h5 file exists, append')
        h5f = h5py.File(write_h5, 'r+')
    else:
        print('write h5 file')
        h5f = h5py.File(write_h5, 'w')
    #%%
    id2s = set([h for h in h5f.keys() if args.training_scheme in h])
    print(len(id2s), l)
    for iteration, data in enumerate(loader, start=start_iter):
        if len(id2s)>=l:
            break
        # torch clear cache
        # torch.cuda.empty_cache()
        # if start_iter is not 0, the 'if' statement will work
        if iteration>=len(loader): break
        log_scalars = {}
        loss_dict = {}
        img_dict = {}
        loss = 0
        model.train()
        iteration += 1
        t1 = default_timer()
        optim.zero_grad()
        fixed, moving = data['voxel1'], data['voxel2']
        fixed = fixed.cuda()
        moving = moving.cuda()

        # do some augmentation
        seg1 = data['segmentation1'].cuda()
        seg2 = data['segmentation2'].cuda()

        id1 = data['id1']
        id2 = data['id2']

        with torch.no_grad():
            warped, flows, agg_flows = s1_model(fixed, moving, return_affine=False)
            sim = similarity(fixed, warped[-1])
            selected = (sim>0.83)
        
        id2 = np.array(id2)
        id1 = np.array(id1)
        selected=  selected.squeeze().cpu().nonzero().squeeze(1).numpy()
        # print(selected, id2)

        id2 = id2[selected]
        sim = sim.squeeze().cpu().numpy()[selected]
        # print(id2)
        h5_keys = list(h5f.keys())
        selected = np.array([i for i in range(len(id2)) if id2[i] not in h5f.keys()])
        if len(selected)==0: continue
        
        id1 = id1[selected]
        id2 = id2[selected]
        fixed = fixed[selected]
        moving = moving[selected]
        print(selected, id2, sim[selected])

        # computed mask: generate mask for tumor region according to flow
        if args.masked in ['soft', 'hard']:
            input_seg, compute_mask = model.pre_register(fixed, moving, None, training=False, cfg=args)
        
        id2s = id2s.union(set(id2))
        # tq.update(len(id2))
        print(len(id2s))
        for i in range(len(id2)):
            h5f.create_group(id2[i])
            h5f[id2[i]].create_dataset('input_seg', data=input_seg[i].cpu().numpy(), dtype='float32')
            h5f[id2[i]].create_dataset('compute_mask', data=compute_mask[i].cpu().numpy(), dtype='float32')
        continue
    #%%
    h5f.close()
    print(len(id2s))
