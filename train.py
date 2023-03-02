from pathlib import Path as pa
import traceback
import h5py
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
from metrics.losses import det_loss, focal_loss, jacobian_det, masked_sim_loss, ortho_loss, reg_loss, dice_jaccard, sim_loss, surf_loss, dice_loss
import datetime as datetime
from torch.utils.tensorboard import SummaryWriter
# from data_util.ctscan import sample_generator
from data_util.dataset import Data, Split
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from run_utils import build_precompute, read_cfg

parser = argparse.ArgumentParser()
parser.add_argument('-bs', "--batch_size", type=int, default=4)
parser.add_argument('-base', '--base_network', type=str, default='VTN')
parser.add_argument('-n', "--n_cascades", type=int, default=3)
parser.add_argument('-e', "--epochs", type=int, default=5)
parser.add_argument("-r", "--round", type=int, default=20000)
parser.add_argument("-v", "--val_steps", type=int, default=1000)
parser.add_argument('-cf', "--checkpoint_frequency", default=0.5, type=float)
parser.add_argument('-c', "--checkpoint", type=lambda x: os.path.realpath(x), default=None)
parser.add_argument('-ct', '--continue_training', action='store_true')
parser.add_argument('--ctt', '--continue_training_this', type=lambda x: os.path.realpath(x), default=None)
parser.add_argument('-g', '--gpu', type=str, default='', help='GPU to use')
parser.add_argument('-d', '--dataset', type=str, default='datasets/liver_cust.json', help='Specifies a data config')
parser.add_argument("-ts", "--training_scheme", type=lambda x:int(x) if x.isdigit() else x, default='', help='Specifies a training scheme')
parser.add_argument('-vs', '--val_scheme', type=lambda x:int(x) if x.isdigit() else x, default='', help='Specifies a validation scheme')
parser.add_argument('--debug', action='store_true', help="run the script without saving files")
parser.add_argument('--name', type=str, default='')

parser.add_argument('--lr_scheduler', default = 'step', type=str, help='lr scheduler', choices=['linear', 'step', 'cosine'])
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('-aug', '--augment', default=False, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='Augment data')
parser.add_argument('-ic', '--in_channel', default=2, type=int, help='Input channel number')
# losses
parser.add_argument('--ortho', type=float, default=0.1, help="use ortho loss")
parser.add_argument('--det', type=float, default=0.1, help="use det loss")
parser.add_argument('--reg', type=float, default=1, help="use reg loss")
parser.add_argument('-inv', '--invert_loss', action='store_true', help="invertibility loss")
parser.add_argument('--surf_loss', default=0, type=float, help='Surface loss weight')
parser.add_argument('-dc', '--dice_loss', default=0, type=float, help='Dice loss weight')
# for stage1 masks
parser.add_argument('-m', '--masked', choices=['soft', 'seg', 'hard'], default='',
     help="mask the tumor part when calculating similarity loss")
parser.add_argument('-mt', '--mask_threshold', type=float, default=1.5, help="volume changing threshold for mask")
parser.add_argument('-mn', '--masked_neighbor', type=int, default=3, help="for masked neibor calculation")
parser.add_argument('-trsf', '--soft_transform', default='sigm', type=str, help='Soft transform')
parser.add_argument('-bnd_thick', '--boundary_thickness', default=0.5, type=float, help='Boundary thickness')
parser.add_argument('-use2', '--use_2nd_flow', default=False, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='Use 2nd flow')
parser.add_argument('-pc', '--pre_calc', default=False, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='Pre-calculate the flow')
parser.add_argument('-s1r', '--stage1_rev', type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=False, help="whether to use reverse flow in stage 1")
parser.add_argument('-os', '--only_shrink', type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="whether to only use shrinkage in stage 1")
# for VP loss
parser.add_argument('-vp', '--vol_preserve', type=float, default=0, help="use volume-preserving loss")
parser.add_argument('-st', '--size_type', choices=['organ', 'tumor', 'tumor_gt', 'constant', 'dynamic', 'reg'], default='tumor', help = 'organ means VP works on whole organ, tumor means VP works on tumor region, tumor_gt means VP works on tumor region with ground truth, constant means VP ratio is a constant, dynamic means VP has dynamic weight, reg means VP is replaced by reg loss')
parser.add_argument('--ks_norm', default='voxel', choices=['image', 'voxel'])
parser.add_argument('-w_ksv', '--w_ks_voxel', default=1, type=float, help='Weight for voxel method in ks loss')
# network structure
parser.add_argument('-ua', '--use_affine', type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="whether to use affine transformation")

# mask calculation needs an extra mask as input
parser.set_defaults(in_channel=3 if parser.parse_args().masked else 2)
parser.set_defaults(stage1_rev=True if parser.parse_args().base_network == 'VXM' else False)
parser.set_defaults(n_cascades=1 if parser.parse_args().base_network != 'VTN' else 3)
parser.set_defaults(use_affine=0 if parser.parse_args().base_network == 'DMR' else 1)

args = parser.parse_args()
# if args.gpu:
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
# set gpu to the one with most free memory
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

def main():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        local_rank = 0

    train_scheme = args.training_scheme or Split.TRAIN
    val_scheme = args.val_scheme or Split.VALID
    print('Train', train_scheme)
    print('Val', val_scheme)
    train_dataset = Data(args.dataset, rounds=args.round*args.batch_size, scheme=train_scheme)
    val_dataset = Data(args.dataset, scheme=val_scheme)
    data_type = 'liver' if 'liver' in args.dataset else 'brain'
    ckp_freq = int(args.checkpoint_frequency * args.round)
    args.data_type = data_type

    # Hong Kong time
    dt = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
    run_id = '_'.join([dt.strftime('%b%d-%H%M%S'), 
                       str(data_type)[:2] + str(train_scheme) + ('pc' if args.pre_calc else ''), # what dataset to use and whether to pre-calculate the flow
                       args.base_network+'x'+str(args.n_cascades), # base network and number of cascades
                       args.name, 
                       args.masked + (f'thr{args.mask_threshold}' + args.soft_transform+'bnd'+str(args.boundary_thickness)+'st{}'.format(1+args.use_2nd_flow) 
                       if args.mask_threshold>0 and args.masked in ['soft', 'hard']  
                       else ''), # params for transforming jacobian
                       'vp'+str(args.vol_preserve)+'st'+args.size_type if args.vol_preserve>0 else '' # params for volume preserving loss
                       ])
    # mkdirs for log and Tensorboard
    if not args.debug:
        if not args.ctt:
            ### TODO: Add the wandb logger
            # wandb.init(project='RCN', config=args, sync_tensorboard=True)
            print("Creating log dir")
            log_dir = os.path.join('./logs', data_type, args.base_network, str(train_scheme), run_id)
            os.path.exists(log_dir) or os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)
            if not os.path.exists(log_dir+'/model_wts/'):
                print("Creating ckp dir", log_dir)
                os.makedirs(log_dir+'/model_wts/')
                ckp_dir = log_dir+'/model_wts/'
        else:
            log_dir = args.ctt
            print("Using log dir", log_dir)
            f = pa(log_dir)
            while True:
                if any(['model_wts' in pc.name for pc in f.iterdir()]):
                    run_id = f.name
                    break
                f = f.parent
            ckp_dir = log_dir+'/model_wts/'
            writer = SummaryWriter(log_dir=log_dir)
        # record args
        with open(log_dir+'/args.txt', 'a') as f:
            from pprint import pprint
            pprint(args.__dict__, f)
            command = "python train.py " + (" ".join(["--{} {}".format(k, v) for k, v in args.__dict__.items() if v])
                                                   + "--ctt " + log_dir)
            print("\nRunning command:\n" + command, file=f)
            # also save the code to the log dir
            import shutil
            shutil.copyfile(os.path.realpath(__file__), log_dir+'/training_code.py')
        args.log_dir = log_dir
        args.run_id = run_id
        print('using run_id:', run_id)
    if args.debug:
        # torch.autograd.set_detect_anomaly(True)
        pass

    # read config
    with open(args.dataset, 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        segmentation_class_value=cfg.get('segmentation_class_value', {'unknown':1})

    in_channels = args.in_channel
    model = RecursiveCascadeNetwork(n_cascades=args.n_cascades, im_size=image_size, base_network=args.base_network, in_channels=in_channels, use_affine=args.use_affine).cuda()
    # compute model size
    total_params = sum(p.nelement()*p.element_size() for p in model.parameters())
    total_buff = sum(b.nelement()*b.element_size() for b in model.buffers())
    size_all_mb = (total_params + total_buff)/ 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))
    
    if args.masked in ['soft', 'hard']:
        if args.pre_calc:
            precompute_h5 = '/home/hynx/regis/recursive-cascaded-networks/datasets/{}_computed.h5'.format(args.base_network)
            assert os.path.exists(precompute_h5), 'Pre-computed flow not found'
            precompute_h5 = h5py.File(precompute_h5, 'r')
            train_dataset.precompute = precompute_h5
            val_dataset.precompute = precompute_h5
        else:
            precompute_h5 = False
            build_precompute(model, train_dataset, args)
        
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        mmodel = model.module
    else: mmodel = model
    trainable_params = []
    for submodel in mmodel.stems:
        trainable_params += list(submodel.parameters())
    trainable_params += list(mmodel.reconstruction.parameters())

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
    if args.checkpoint or args.ctt:
        ckp = args.checkpoint or ckp_dir
        if os.path.isdir(ckp):
            ckp = load_model_from_dir(ckp, model)
        else: load_model(torch.load(os.path.join(ckp)), model)
        print("Loaded checkpoint from {}".format(ckp))
        if args.continue_training or args.ctt:
            state = torch.load(ckp)
            start_epoch = state['epoch']
            start_iter = state['global_iter'] + 1
            optim_state = state['optimizer_state_dict']
            optim.load_state_dict(optim_state)
            scheduler_state = state['scheduler_state_dict']
            scheduler.load_state_dict(scheduler_state)
            print("Continue training from checkpoint from epoch {} iter {}".format(start_epoch, start_iter))
    num_worker = min(8, args.batch_size)
    if dist.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_worker, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_worker, shuffle=True)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=num_worker, shuffle=False)

    # Saving the losses
    train_loss_log = []
    reg_loss_log = []
    val_loss_log = []
    max_seg = max(segmentation_class_value.values())

    # use cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(3749)
    for epoch in range(start_epoch, args.epochs):
        print(f"-----Epoch {epoch+1} / {args.epochs}-----")
        train_epoch_loss = 0
        train_reg_loss = 0
        vis_batch = []
        t0 = default_timer()
        st_t = default_timer()
        for iteration, data in enumerate(train_loader, start=start_iter):
            # torch clear cache
            # torch.cuda.empty_cache()
            # if start_iter is not 0, the 'if' statement will work
            if iteration>=len(train_loader): break
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
            
            # computed mask: generate mask for tumor region according to flow
            if args.masked in ['soft', 'hard']:
                if precompute_h5:
                    input_seg, compute_mask = data['input_seg'].cuda(), data['compute_mask'].cuda()
                else:
                    input_seg, compute_mask, ls, idct = model.pre_register(fixed, moving, seg2, training=True, cfg=args)
                    log_scalars.update(ls)
                    img_dict.update(idct)
            if args.masked=='seg': input_seg = seg2.float()
            
            ## No augment here
            # if args.augment and (not args.debug):
            #     moving, seg2 = model.augment(moving, seg2)
            if args.invert_loss:
                fixed, moving = torch.cat([fixed, moving], dim=0), torch.cat([moving, fixed], dim=0)
                seg1, seg2 = torch.cat([seg1, seg2], dim=0), torch.cat([seg2, seg1], dim=0)
            
            if args.in_channel==3:
                moving_ = torch.cat([moving, input_seg], dim=1)
            else:
                moving_ = moving

            returns = model(fixed, moving_, return_affine=args.use_affine)
            if args.use_affine:
                warped_, flows, agg_flows, affine_params = returns
            else:
                warped_, flows, agg_flows = returns
            warped = [i[:, :-1] for i in warped_] if args.in_channel>2 else warped_

            # affine loss
            if args.use_affine:
                ortho_factor, det_factor = args.ortho, args.det
                A = affine_params['theta'][..., :3, :3]
                ort = ortho_factor * ortho_loss(A)
                det = det_factor * det_loss(A)
                ort, det = reduce_mean(ort), reduce_mean(det)
                loss_dict['ortho'] = ort.item()
                loss_dict['det'] = det.item()
                loss = loss + ort + det
            with torch.no_grad():
                # w_seg2 is always float type
                w_seg2 = mmodel.reconstruction(seg2.float(), agg_flows[-1])
                # warped_tseg represents the tumor region in the warped image, boolean type
                warped_tseg = w_seg2>1.5
                # w_oseg represents the organ region in the warped image, float type
                w_oseg = mmodel.reconstruction((seg2>0.5).float(), agg_flows[-1])
                # add log scalar tumor/ratio
                if (seg2>1.5).sum().item() > 0:
                    t_c = (warped_tseg.sum(dim=(1,2,3,4)).float() / (seg2 > 1.5).sum(dim=(1,2,3,4)).float())
                    t_c_nn = t_c[t_c.isnan()==False]
                    log_scalars['tumor_change'] = t_c_nn.mean().item()
                    o_c = (w_seg2>0.5).sum(dim=(1,2,3,4)).float() / (seg2 > 0.5).sum(dim=(1,2,3,4)).float()
                    o_c_nn = o_c[t_c.isnan()==False]
                    log_scalars['organ_change'] = o_c_nn.mean().item()
                    log_scalars['to_ratio'] = torch.where(t_c_nn/o_c_nn >1, t_c_nn/o_c_nn, o_c_nn/t_c_nn).mean().item()

                # add log scalar dice_organ
                dice_organ, _ = dice_jaccard(w_seg2>0.5, seg1>0.5)
                log_scalars['dice_organ'] = dice_organ.mean().item()

            # reg loss
            reg = reg_loss(
                flows[int(args.use_affine):]
                ) * args.reg
            # default masks used for VP loss
            mask = warped_tseg | (seg1>1.5)
            mask_moving = seg2
            vp_seg_mask = w_seg2
            # sim loss
            if not args.masked:
                sim = sim_loss(fixed, warped[-1])
            else:
                # setup the vp_seg_mask for VP loss
                assert mask.requires_grad == False
                if args.masked == 'seg':
                    sim = masked_sim_loss(fixed, warped[-1], mask)
                elif args.masked == 'soft':
                    # soft mask * sim loss per pixel, soft mask should be computed from stage1 model
                    with torch.no_grad():
                        warped_soft_mask = mmodel.reconstruction(compute_mask.float(), agg_flows[-1])
                        warped_mask = mmodel.reconstruction(input_seg.float(), agg_flows[-1])
                    sim = sim_loss(fixed, warped[-1], 1-warped_soft_mask)
                    # vp_seg_mask now is a continuous mask that represents how shrinky the voxel is
                    vp_seg_mask = warped_mask
                    if args.debug:
                        # import code; code.interact(local=dict(globals(), **locals()))
                        # calculate the strenght of soft contraints compared to hard constraints
                        (warped_soft_mask>0.5).flatten(1).sum(1)/((w_seg2>1.5).flatten(1).sum(1)+1e-6)
                        # calculate the strenght of soft contraints 
                        (warped_soft_mask)[w_seg2>1.5].sum()/((w_seg2>1.5).sum()+1e-6)
                elif args.masked =='hard':
                    with torch.no_grad():
                        warped_hard_mask = mmodel.reconstruction(compute_mask.float(), agg_flows[-1]) > 0.5
                        warped_mask = mmodel.reconstruction(input_seg.float(), agg_flows[-1])
                    # TODO: may consider mask outside organ regions
                    sim = masked_sim_loss(fixed, warped[-1], warped_hard_mask)
                    vp_seg_mask = warped_mask

            if args.dice_loss>0:
                # using vanilla dice loss
                # dice_lo = dice_loss(w_oseg, (seg1>0.5).float()) * args.dice_loss
                # using dice loss that emphasizes boundary
                # dice_lo = dice_loss(w_oseg, (seg1>0.5).float(), sharpen=0.9) * args.dice_loss
                # using focal loss to emphasize boundary
                dice_lo = focal_loss(w_oseg, (seg1>0.5).float(), alpha=0.25, gamma=3) * args.dice_loss
                loss_dict.update({'dice_loss': dice_lo.item()})
                loss = loss + dice_lo

            # VP loss
            # if args.vol_preserve>0 and args.size_type=='reg':
            #     vp_tum_loc = vp_seg_mask>0.5
            #     def reg_mask(flow, mask):
            #         deno = mask.sum(dim=(1,2,3,4)).float()
            #         dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
            #         dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
            #         dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
            #         dy = dy * mask[:, :, 1:, :, :]
            #         dx = dx * mask[:, :, :, 1:, :]
            #         dz = dz * mask[:, :, :, :, 1:]
            #         d = (dy**2).sum(dim=(1,2,3,4))/deno + (dx**2).sum(dim=(1,2,3,4))/deno + (dz**2).sum(dim=(1,2,3,4))/deno
            #         return d.mean()/2.0 *4.0
            #     k_sz = reg_mask(agg_flows[-1], vp_tum_loc) * args.vol_preserve

            if args.vol_preserve>0:
                # to decide the voxels to be preserved
                if args.size_type=='organ':
                    # keep the volume of the whole organ 
                    vp_tum_loc = vp_seg_mask>0.5
                elif args.size_type=='tumor': 
                    # keep the volume of the tumor region
                    vp_tum_loc = vp_seg_mask>1.5
                elif args.size_type=='tumor_gt':
                    # use warped gt mask to know exactly the location of tumors
                    vp_tum_loc = w_seg2>1.5
                    vp_seg_mask = w_seg2
                elif args.size_type=='dynamic':
                    # the vp_loc is a weighted mask that is calculated from the ratio of the tumor region
                    vp_tum_loc = warped_soft_mask # B, C, S, H, W
                img_dict['vp_loc'] = visualize_3d(vp_tum_loc[0,0]).cpu()
                img_dict['vp_loc_on_wseg2'] = visualize_3d(draw_seg_on_vol(vp_tum_loc[0,0], 
                                                                           w_seg2[0,0].round().long(), 
                                                                           to_onehot=True)).cpu()
                bs = agg_flows[-1].shape[0]
                # the following line uses mask_fixing and mask_moving to calculate the ratio
                # ratio = (mask_fixing>.5).view(bs,-1).sum(1)/(mask_moving>.5).view(bs, -1).sum(1); 
                # now I try to use the warped mask as the reference
                ratio = (vp_seg_mask>0.5).view(bs,-1).sum(1) / (mask_moving>0.5).view(bs, -1).sum(1)
                log_scalars['target_warped_ratio'] = ratio.mean().item()
                ratio=ratio.view(-1,1,1,1)
                k_sz = 0
                ### single voxel ratio
                if args.w_ks_voxel>0:
                    # Notice!! It should be *ratio instead of /ratio
                    det_flow = (jacobian_det(agg_flows[-1], return_det=True).abs()*ratio).clamp(min=1/3, max=3)
                    vp_mask = F.interpolate(vp_tum_loc.float(), size=det_flow.shape[-3:], mode='trilinear', align_corners=False).float().squeeze() # B, S, H, W
                    with torch.no_grad():  
                        vp_seg = F.interpolate(w_seg2, size=det_flow.shape[-3:], mode='trilinear', align_corners=False).float().squeeze().round().long() # B, S, H, W
                        img_dict['det_flow'] = visualize_3d(det_flow[0]).cpu()
                    if args.ks_norm=='voxel':
                        adet_flow = torch.where(det_flow>1, det_flow, (1/det_flow))
                        k_sz_voxel = (adet_flow*vp_mask).sum()/vp_mask.sum()
                        img_dict['vp_det_flow'] = visualize_3d(adet_flow[0]).cpu()
                        img_dict['vpdetflow_on_seg2'] = visualize_3d(draw_seg_on_vol(adet_flow[0], vp_seg[0], to_onehot=True)).cpu()
                    elif args.ks_norm=='image':
                        # normalize loss for every image
                        k_sz_voxel = (torch.where(det_flow>1, det_flow, 1/det_flow)*(vp_mask)).sum(dim=(-1,-2,-3))
                        nonzero_idx = k_sz_voxel.nonzero()
                        k_sz_voxel = k_sz_voxel[nonzero_idx]/vp_mask[:,0][nonzero_idx].sum(dim=(-1,-2,-3))
                        k_sz_voxel = k_sz_voxel.mean()
                    else: raise NotImplementedError
                    k_sz = k_sz + args.w_ks_voxel*k_sz_voxel
                    log_scalars['k_sz_voxel'] = k_sz_voxel.item()
                ### whole volume ratio
                if args.w_ks_voxel<1:
                    det_flow = jacobian_det(agg_flows[-1], return_det=True).abs()
                    vp_mask = F.interpolate(vp_tum_loc.float(), size=det_flow.shape[-3:], mode='trilinear', align_corners=False) > 0.5
                    det_flow[~vp_mask[:,0]]=0
                    k_sz_volume = (det_flow.sum(dim=(-1,-2,-3))/vp_mask[:,0].sum(dim=(-1,-2,-3))/ratio.squeeze()).clamp(1/3, 3)
                    k_sz_volume = torch.where(k_sz_volume>1, k_sz_volume, 1/k_sz_volume)
                    k_sz_volume = k_sz_volume[k_sz_volume.nonzero()].mean()
                    k_sz = k_sz + (1-args.w_ks_voxel)*k_sz_volume
                    log_scalars['k_sz_volume'] = k_sz_volume.item()
                k_sz = k_sz  * args.vol_preserve
                if torch.isnan(k_sz): k_sz = sim.new_zeros(1) # k_sz is all nan
                k_sz = reduce_mean(k_sz)
                loss_dict['vol_preserve_loss'] = k_sz.item()
                loss = loss + k_sz
            
            if args.invert_loss:
                foward_flow, backward_flow = agg_flows[-1][:args.batch_size], agg_flows[-1][args.batch_size:]
                f12, f21 = mmodel.reconstruction(backward_flow, foward_flow)+foward_flow, mmodel.reconstruction(foward_flow, backward_flow)+backward_flow
                f12_d, f21_d = (f12**2).sum(1), (f21**2).sum(1)
                thres12, thres21 = torch.quantile(f12_d, .8), torch.quantile(f21_d, .8)
                f12_d, f21_d = f12_d.clamp(max=thres12), f21_d.clamp(max=thres21)
                ivt = (f12_d.mean()+f21_d.mean())/2
                ivt = ivt*0.1
                loss_dict.update({'invert_loss':ivt.item()})
                loss = loss + ivt

            if args.surf_loss:
                surf = surf_loss(w_seg2>0.5, seg1>0.5, agg_flows[-1]) * args.surf_loss
                loss_dict.update({'surf_loss':surf.item()})
                loss = loss + surf

            loss = loss+sim+reg
            if loss.isnan().any():
                import ipdb; ipdb.set_trace()
                loss = 0
            loss.backward()
            optim.step()
            # ddp reduce of loss
            loss, sim, reg = reduce_mean(loss), reduce_mean(sim), reduce_mean(reg)
            loss_dict.update({
                'sim_loss': sim.item(),
                'reg_loss': reg.item(),
                'loss': loss.item()
            })

            train_epoch_loss = train_epoch_loss + loss.item()
            train_reg_loss = train_reg_loss + reg.item()

            if local_rank==0 and (iteration%10==0):
                if iteration<500 or iteration % 500 == 0:
                    avg_time_per_iter = (default_timer() - st_t) / (iteration + 1 - start_iter)
                    est_time_for_epo = avg_time_per_iter * (len(train_loader) - iteration)
                    est_time_for_train = (args.epochs - epoch - 1) * len(train_loader) * avg_time_per_iter + est_time_for_epo
                    log_scalars.update({
                        'est_remain_time': est_time_for_train
                    })
                    print('*%s* ' % run_id,
                          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                          'Epoch %d Steps %d, Total time %.2f, data %.2f%%. Loss %.3e lr %.3e' % (epoch, iteration,
                                                                                         default_timer() - t0,
                                                                                         (t1 - t0) / (
                                                                                             default_timer() - t0),
                                                                                         loss,
                                                                                         lr),
                          end='\n')
                    print('Estimated time for epoch (H/M/S): %s, Estimated time for training (H/M/S): %s' % (
                        str(datetime.timedelta(seconds=est_time_for_epo)),
                        str(datetime.timedelta(seconds=est_time_for_train))),
                          end='\r')
                    if not args.debug:
                        writer.add_image('train/img1', visualize_3d(fixed[0, 0]), epoch * len(train_loader) + iteration)
                        writer.add_image('train/img2', visualize_3d(moving[0, 0]), epoch * len(train_loader) + iteration)
                        writer.add_image('train/warped', visualize_3d(warped[-1][0, 0]), epoch * len(train_loader) + iteration)
                        writer.add_image('train/seg1', visualize_3d(seg1[0, 0]), epoch * len(train_loader) + iteration)
                        writer.add_image('train/seg2', visualize_3d(seg2[0, 0]), epoch * len(train_loader) + iteration)
                        writer.add_image('train/w_seg2', visualize_3d(w_seg2[0, 0]), epoch * len(train_loader) + iteration)
                        for k,v in img_dict.items():
                            writer.add_image('train/'+k, v, epoch * len(train_loader) + iteration)

                if not args.debug:
                    for k in loss_dict:
                        writer.add_scalar('train/'+k, loss_dict[k], epoch * len(train_loader) + iteration)
                    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch * len(train_loader) + iteration)
                    for k in log_scalars:
                        writer.add_scalar('train/'+k, log_scalars[k], epoch * len(train_loader) + iteration)

                if args.val_steps>0  and (iteration%args.val_steps==0 or args.debug):
                    print(f">>>>> Validation <<<<<")
                    val_epoch_loss = 0
                    dice_loss_val = {k:0 for k in segmentation_class_value.keys()}
                    to_ratios, total_to = 0,0
                    model.eval()
                    tb_imgs = {}
                    for itern, data in enumerate(val_loader):
                        itern += 1
                        if len(val_loader)>3 and itern % int(0.33 * len(val_loader)) == 0:
                            print(f"\t-----Iteration {itern} / {len(val_loader)} -----")
                        with torch.no_grad():
                            fixed, moving = data['voxel1'], data['voxel2']
                            seg2 = data['segmentation2'].cuda()
                            fixed = fixed.cuda()
                            moving = moving.cuda()
                            if args.masked=='seg':
                                moving_ = torch.cat([moving, seg2], dim=1)
                            elif args.masked in ['soft', 'hard']:
                                if precompute_h5:
                                    input_seg, compute_mask = data['input_seg'].cuda(), data['compute_mask'].cuda()
                                else:
                                    input_seg, compute_mask = model.pre_register(fixed, moving, seg2, training=False, cfg=args)
                                moving_ = torch.cat([moving, input_seg], dim=1)
                            else:
                                moving_ = moving
                            warped_, flows, agg_flows = mmodel(fixed, moving_)
                            warped = [i[:, :1] for i in warped_]
                            # w_seg2 is the warped gt mask of moving
                            w_seg2 = model.reconstruction(seg2.float().cuda(), agg_flows[-1].float())
                            sim, reg = sim_loss(fixed, warped[-1]), reg_loss(flows[1:])
                            loss = sim + reg
                            val_epoch_loss += loss.item()
                            for k,v in segmentation_class_value.items():
                                # Single category Segmentation map
                                sseg1 = data['segmentation1'].cuda() > (v-0.5)
                                w_sseg2 = w_seg2 > (v-0.5)
                                dice, jac = dice_jaccard(sseg1, w_sseg2)
                                dice_loss_val[k] += dice.mean().item()
                            # add metrics for to_ratio
                            to_ratio = (torch.sum(w_seg2>1.5, dim=(2, 3, 4)) / torch.sum(seg2>1.5, dim=(2, 3, 4))) /(torch.sum(w_seg2>0.5, dim=(2, 3, 4)) / torch.sum(seg2>0.5, dim=(2, 3, 4)))
                            to_ratio = torch.where(to_ratio>1, to_ratio, 1/to_ratio)
                            # only select those that are not nan or inf
                            to_ratio = to_ratio[to_ratio.isfinite() & (~to_ratio.isnan())]
                            to_ratios += to_ratio.sum()
                            total_to += len(to_ratio)

                    tb_imgs['img1'] = fixed
                    tb_imgs['img2'] = moving
                    tb_imgs['warped'] = warped[-1]
                    with torch.no_grad():
                        tb_imgs['seg1'] = data['segmentation1']
                        tb_imgs['seg2'] = data['segmentation2']
                        tb_imgs['w_seg2'] = w_seg2

                    mean_val_loss = val_epoch_loss / len(val_loader)
                    print(f"Mean val loss: {mean_val_loss}")
                    val_loss_log.append(mean_val_loss)
                    mean_dice_loss = {}
                    for k, v in dice_loss_val.items():
                        mean_dc = v / len(val_loader)
                        mean_dice_loss[k] = mean_dc
                        print(f'Mean dice loss {k}: {mean_dc}')
                    mean_to_ratio = total_to and to_ratios / total_to
                    print(f'Mean to_ratio: {mean_to_ratio}')
                    if not args.debug:
                        writer.add_scalar('val/loss', mean_val_loss, epoch * len(train_loader) + iteration)
                        writer.add_scalar('val/to_ratio', mean_to_ratio, epoch * len(train_loader) + iteration)
                        for k in mean_dice_loss.keys():
                            writer.add_scalar(f'val/dice_{k}', mean_dice_loss[k], epoch * len(train_loader) + iteration)
                        # add img1, img2, seg1, seg2, warped, w_seg2
                        for k, im in tb_imgs.items():
                            writer.add_image(f'val/{k}', visualize_3d(im[0,0]), epoch * len(train_loader) + iteration)

            if local_rank==0 and iteration % ckp_freq == 0:
                if not args.debug and not os.path.exists('./ckp/model_wts/'+run_id):
                    os.makedirs('./ckp/model_wts/'+run_id)
                
                train_loss_log.append(train_epoch_loss / iteration)
                reg_loss_log.append(train_reg_loss / iteration)

                ckp = {}
                ckp['stem_state_dict'] = mmodel.stems.state_dict()

                ckp['train_loss'] = train_loss_log
                ckp['val_loss'] = val_loss_log
                ckp['epoch'] = epoch
                ckp['global_iter'] = iteration
                #TODO: add optimizer state dict
                ckp['optimizer_state_dict'] = optim.state_dict()
                ckp['scheduler_state_dict'] = scheduler.state_dict()

                torch.save(ckp, f'{ckp_dir}/epoch_{epoch}_iter_{iteration}.pth')

            del fixed, moving, seg2, warped_, flows, agg_flows, loss, sim, reg, 
            del loss_dict, log_scalars, data, img_dict
            t0 = default_timer()   
        scheduler.step()
        start_iter = 0

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # rename log_dir
        log_dir = args.log_dir
        print('training interrupted')
        print(log_dir)
        # os.rename(log_dir, log_dir+'_interrupted')
    except Exception as e:
        # rename log_dir
        log_dir = args.log_dir
        print(log_dir)
        # os.rename(log_dir, log_dir+'_error')
        raise e