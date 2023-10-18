import ml_collections
import hashlib
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
from data_util.dataset import Data, Split
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from run_utils import build_precompute, read_cfg

parser = argparse.ArgumentParser()

# Training settings
parser.add_argument('-bs', "--batch_size", type=int, default=4)
parser.add_argument('-e', "--epochs", type=int, default=5)
parser.add_argument("-r", "--round", type=int, default=20000)
parser.add_argument("-v", "--val_steps", type=int, default=1000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument('--lr_scheduler', default='step', type=str, choices=['linear', 'step', 'cosine'], help='lr scheduler')
parser.add_argument('-aug', '--augment', type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=False, help='Augment data')
parser.add_argument('--training_scheme', type=lambda x:int(x) if x.isdigit() else x, default='', help='Specifies a training scheme')
parser.add_argument('-vs', '--val_scheme', type=lambda x:int(x) if x.isdigit() else x, default='', help='Specifies a validation scheme')
parser.add_argument('--debug', action='store_true', help="run the script without saving files")

# Checkpoint settings
parser.add_argument('-cf', "--checkpoint_frequency", type=float, default=0.1)
parser.add_argument('-c', "--checkpoint", type=lambda x: os.path.realpath(x), default=None)
parser.add_argument('-ct', '--continue_training', action='store_true')
parser.add_argument('--ctt', '--continue_training_this', type=lambda x: os.path.realpath(x), default=None)

# Dataset settings
parser.add_argument('-d', '--dataset', type=str, default='datasets/liver_cust.json', help='Specifies a data config')
parser.add_argument('-ic', '--in_channel', type=int, default=2, help='Input channel number')
parser.add_argument('-g', '--gpu', type=str, default='', help='GPU to use')
parser.add_argument('--name', type=str, default='')

# Regular loss settings
parser.add_argument('--ortho', type=float, default=0.1, help="use ortho loss")
parser.add_argument('--det', type=float, default=0.1, help="use det loss")
parser.add_argument('--reg', type=float, default=1, help="use reg loss")

# Network structure settings
parser.add_argument('-base', '--base_network', type=str, default='VTN')
parser.add_argument('-n', "--n_cascades", type=int, default=3)
parser.add_argument('-ua', '--use_affine', type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="whether to use affine transformation")

# Mask and transformation settings
parser.add_argument('-m', '--masked', choices=['soft', 'hard'], default='', help="mask the tumor part when calculating similarity loss")
parser.add_argument('-msd', '--mask_seg_dice', type=float, default=-1, help="choose the accuracy of seg mask when using seg mask")
parser.add_argument('-mt', '--mask_threshold', type=float, default=1.5, help="volume changing threshold for mask")
parser.add_argument('-mn', '--masked_neighbor', type=int, default=3, help="for masked neighbor calculation")
parser.add_argument('-trsf', '--soft_transform', type=str, default='sigm', help='Soft transform')
parser.add_argument('-bnd_thick', '--boundary_thickness', type=float, default=0.5, help='Boundary thickness')
parser.add_argument('-use2', '--use_2nd_flow', type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=False, help='Use 2nd flow')
parser.add_argument('-useb', '--use_bilateral', type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help='Use bilateral filter')
parser.add_argument('-os', '--only_shrink', type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="whether to only use shrinkage in stage 1")

# Volume preserving loss settings
parser.add_argument('-vp', '--vol_preserve', type=float, default=0, help="use volume-preserving loss")
parser.add_argument('-st', '--size_type', choices=['organ', 'tumor', 'dynamic'], default='tumor', help='organ means VP works on whole organ, tumor means VP works on tumor region, dynamic means VP has dynamic weight')
parser.add_argument('--ks_norm', default='voxel', choices=['image', 'voxel'])
parser.add_argument('-w_ksv', '--w_ks_voxel', type=float, default=1, help='Weight for using soft mask method in ks loss')

# Hyper VP setting (Ignored, under development)
parser.add_argument('-hpv', '--hyper_vp', action='store_true', help="whether to use hypernet for VP")

# Default settings based on other arguments
parser.set_defaults(in_channel=3 if parser.parse_args().masked else 2)
parser.set_defaults(n_cascades=1 if parser.parse_args().base_network != 'VTN' else 3)
parser.set_defaults(use_affine=0 if parser.parse_args().base_network == 'DMR' else 1)

def reduce_mean(tensor):
    if dist.is_initialized():
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= dist.get_world_size()
        return rt
    else:
        return tensor

def hyp_param_generator(param_range, oversample=.2, batch=1):
    rand_gen = torch.rand(batch)
    hyp_param = torch.zeros(batch)
    for i in range(batch):
        if rand_gen[i] < oversample:
            # choose param[0] or param[-1]
            hyp_param[i] = param_range[0] if torch.rand(1) < 0.5 else param_range[-1]
        else:
            hyp_param[i] = torch.rand(1) * (param_range[-1] - param_range[0]) + param_range[0]
    return hyp_param[:,None]

def main(args):
    train_scheme = args.training_scheme or Split.TRAIN
    val_scheme = args.val_scheme or Split.VALID
    print('Train', train_scheme)
    print('Val', val_scheme)
    train_dataset = Data(args.dataset, rounds=args.round*args.batch_size, scheme=train_scheme)
    val_dataset = Data(args.dataset, scheme=val_scheme)
    data_type = 'liver' if 'liver' in args.dataset else 'brain'
    ckp_freq = int(args.checkpoint_frequency * args.round)
    args.data_type = data_type

    # get time and generate run_id
    dt = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8))) # HK time
    run_id = '_'.join([dt.strftime('%b%d-%H%M%S'),
                       str(data_type)[:2] + str(train_scheme), # what dataset to use
                       args.base_network+'x'+str(args.n_cascades), # base network and number of cascades
                       args.name,
                       args.masked + (f'thr{args.mask_threshold}' + args.soft_transform+'bnd'+str(args.boundary_thickness)+'st{}'.format(1+args.use_2nd_flow) + ('bf' if args.use_bilateral and args.use_2nd_flow else '')
                       if args.mask_threshold>0 and args.masked in ['soft', 'hard']
                       else ''), # params for transforming jacobian
                       'vp'+str(args.vol_preserve)+'st'+args.size_type if args.vol_preserve>0 else '' # params for volume preserving loss
                       ])
    print('Current Run ID:', run_id)
    # mkdirs for log and Tensorboard
    if not args.debug:
        if not args.ctt:
            print("Creating log dir")
            log_dir = os.path.join('./logs', data_type, args.base_network, 'hyp' if args.hyper_vp else str(train_scheme), run_id)
            os.path.exists(log_dir) or os.makedirs(log_dir)
            if not os.path.exists(log_dir+'/model_wts/'):
                print("Creating ckp dir", os.path.abspath(log_dir))
                os.makedirs(log_dir+'/model_wts/')
                ckp_dir = log_dir+'/model_wts/'
        else:
            log_dir = args.ctt
            print("Using log dir", os.path.abspath(log_dir))
            f = pa(log_dir)
            while True:
                if any(['model_wts' in pc.name for pc in f.iterdir()]):
                    run_id = f.name
                    break
                f = f.parent
            ckp_dir = log_dir+'/model_wts/'
            ## Directly load the args from the log_dir
            cfg = read_cfg(log_dir)
            if cfg.get('log_dir', None) is not None:
                assert cfg['log_dir'] == log_dir
            if cfg.get('run_id', None) is not None:
                assert cfg['run_id'] == run_id
            for k,v in cfg.items():
                if 'training' not in k and 'continue' not in k and 'checkpoint' not in k and 'ctt' not in k:
                    setattr(args, k, v)

        # record args for reuse in evaluation
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

        if args.use_wandb:
            import wandb
            cfg = args.__dict__
            # remove cfg name
            cfg.pop('name')
            name = args.run_id.split('_')[1:]
            name = '_'.join(name)
            # hash run_id into a short string
            id_for_wandb = hashlib.md5(run_id.encode()).hexdigest()[:8]
            tags = run_id.split('_') + [data_type, args.base_network, str(train_scheme)]
            # rm empty tags
            tags = [t for t in tags if t]
            wandb.init(name=name, notes=run_id, sync_tensorboard=True, config=cfg, save_code=True, dir=pa(log_dir).parent,
                       resume='allow' if not args.ctt else 'must', id=id_for_wandb, tags=tags)
        # print in green the following message
        print('\033[92m')
        writer = SummaryWriter(log_dir=log_dir)
        print('\033[0m')

    # read config
    with open(args.dataset, 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        segmentation_class_value=cfg.get('segmentation_class_value', {'unknown':1})

    in_channels = args.in_channel
    model = RecursiveCascadeNetwork(n_cascades=args.n_cascades, im_size=image_size, base_network=args.base_network,
                                    in_channels=in_channels, use_affine=args.use_affine, hyper_net=args.hyper_vp,
                                    ).cuda()

    # compute model size
    total_params = sum(p.nelement()*p.element_size() for p in model.parameters())
    total_buff = sum(b.nelement()*b.element_size() for b in model.buffers())
    size_all_mb = (total_params + total_buff)/ 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

    if args.masked in ['soft', 'hard']:
        cfg_train = ml_collections.ConfigDict(args.__dict__)
        build_precompute(model, train_dataset, cfg_train)

    trainable_params = []
    for submodel in model.stems:
        trainable_params += list(submodel.parameters())
    trainable_params += list(model.reconstruction.parameters())

    # NOTE build optimizer
    lr = args.lr
    optim = Adam(trainable_params, lr=lr, weight_decay=1e-4)
    if args.lr_scheduler == 'step':
        scheduler = StepLR(optimizer=optim, step_size=10, gamma=0.96)
    elif args.lr_scheduler == 'linear':
        min_lr = 1e-6
        scheduler = LambdaLR(optimizer=optim, lr_lambda=lambda epoch: min_lr + (lr - min_lr) * (1 - epoch / args.epochs), last_epoch= start_epoch-1)

    # NOTE load checkpoint
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
            optim_state = torch.load(pa(ckp).parent/'optim.pth')
            optim.load_state_dict(optim_state['optimizer_state_dict'])
            scheduler.load_state_dict(optim_state['scheduler_state_dict'])
            print("Continue training from checkpoint from epoch {} iter {}".format(start_epoch, start_iter))
    # NOTE dataloader
    num_worker = min(8, args.batch_size)
    torch.manual_seed(3749)
    if dist.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_worker, sampler=train_sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_worker, shuffle=True)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=num_worker, shuffle=False)

    train_loss_log = []
    reg_loss_log = []
    val_loss_log = []
    max_seg = max(segmentation_class_value.values())

    # use cudnn.benchmark for deterministic training
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    for epoch in range(start_epoch, args.epochs):
        print(f"-----Epoch {epoch+1} / {args.epochs}-----")
        train_epoch_loss = 0
        train_reg_loss = 0
        vis_batch = []
        t0 = default_timer()
        st_t = default_timer()
        # SECTION training
        for iteration, data in enumerate(train_loader, start=start_iter):
            if iteration>=len(train_loader): break
            log_scalars = {}
            log_list = {}
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

            # NOTE compute mask: generate mask for tumor region according to flow
            if args.masked in ['soft', 'hard']:
                input_seg, compute_mask, ls, idct = model.pre_register(fixed, moving, seg2, training=True, cfg=cfg_train)
                log_scalars.update(ls)
                img_dict.update(idct)

            if args.in_channel==3:
                moving_ = torch.cat([moving, input_seg], dim=1)
            else:
                moving_ = moving

            # NOTE the running of the model
            if args.hyper_vp:
                hyp_param = hyp_param_generator([0, args.vol_preserve], batch=fixed.shape[0]).cuda()
                # might log the list instead of the first one
                log_scalars['hyp_vp'] = hyp_param[0].item()
                returns = model(fixed, moving_, return_affine=args.use_affine, hyp_input=hyp_param)
            else:
                returns = model(fixed, moving_, return_affine=args.use_affine)

            if args.use_affine:
                warped_, flows, agg_flows, affine_params = returns
            else:
                warped_, flows, agg_flows = returns
            warped = [i[:, :-1] for i in warped_] if args.in_channel>2 else warped_

            # NOTE affine loss
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
                w_seg2 = model.reconstruction(seg2.float(), agg_flows[-1])
                warped_tseg = w_seg2>1.5
                w_oseg = w_seg2>0.5

                ######### logging #########
                # add scalar tumor/ratio
                if (seg2>1.5).sum().item() > 0:
                    t_c = (warped_tseg.sum(dim=(1,2,3,4)).float() / (seg2 > 1.5).sum(dim=(1,2,3,4)).float())
                    t_c_nn = t_c[t_c.isnan()==False]
                    log_scalars['tumor_change'] = t_c_nn.mean().item()
                    o_c = (w_seg2>0.5).sum(dim=(1,2,3,4)).float() / (seg2 > 0.5).sum(dim=(1,2,3,4)).float()
                    o_c_nn = o_c[t_c.isnan()==False]
                    log_scalars['organ_change'] = o_c_nn.mean().item()
                    log_scalars['to_ratio'] = torch.where(t_c_nn/o_c_nn >1, t_c_nn/o_c_nn, o_c_nn/t_c_nn).mean().item()
                # add scalar dice_organ
                dice_organ, _ = dice_jaccard(w_seg2>0.5, seg1>0.5)
                log_scalars['dice_organ'] = dice_organ.mean().item()

            # default masks used for VP loss
            # mask_moving: the mask where the ratio is computed, used for VP loss
            # vp_seg_mask: the mask where the ratio is kept, used for VP loss
            sims = []
            if not args.masked:
                sim = sim_loss(fixed, warped[-1])
                mask_moving = seg2
                vp_seg_mask = w_seg2
            else:
                mask_moving = input_seg
                with torch.no_grad():
                    warped_mask = model.reconstruction(input_seg.float(), agg_flows[-1])
                vp_seg_mask = warped_mask
                if args.masked == 'soft':
                    # soft mask * sim loss per pixel, soft mask should be computed from stage1 model
                    with torch.no_grad():
                        warped_soft_mask = model.reconstruction(compute_mask.float(), agg_flows[-1])

                    if args.hyper_vp:
                        hyp_compute_mask = (1-warped_soft_mask)**(hyp_param/args.vol_preserve)[...,None,None,None]
                        sims = sim_loss(fixed, warped[-1], hyp_compute_mask, return_mean=False)
                        sim = sims.mean()
                    else:
                        sim = sim_loss(fixed, warped[-1], 1-warped_soft_mask)
                    vp_seg_mask = warped_mask
                elif args.masked =='hard':
                    with torch.no_grad():
                        warped_hard_mask = model.reconstruction(compute_mask.float(), agg_flows[-1]) > 0.5
                    sims = masked_sim_loss(fixed, warped[-1], warped_hard_mask)
                    sim = sims.mean()
                    vp_seg_mask = warped_mask
            loss = loss + sim

            reg = reg_loss(
                flows[int(args.use_affine):]
                ) * args.reg
            loss = loss + reg

            k_szs = []
            if args.vol_preserve>0:
                # get vp_tum_loc: tum_mask for vp
                if args.size_type=='organ':
                    # keep the volume of the whole organ
                    vp_tum_loc = vp_seg_mask>0.5
                elif args.size_type=='tumor':
                    # keep the volume of the tumor region
                    vp_tum_loc = vp_seg_mask>1.5
                elif args.size_type=='dynamic':
                    # the vp_loc is a weighted mask that is calculated from the ratio of the tumor region
                    vp_tum_loc = warped_soft_mask # B, C, S, H, W
                # calculate the TSR: tumor size ratio
                bs = agg_flows[-1].shape[0]
                ratio = (vp_seg_mask>0.5).view(bs,-1).sum(1) / (mask_moving>0.5).view(bs, -1).sum(1)
                ratio=ratio.view(-1,1,1,1)

                # log
                img_dict['vp_loc'] = visualize_3d(vp_tum_loc[0,0]).cpu()
                img_dict['vp_loc_on_wseg2'] = visualize_3d(draw_seg_on_vol(vp_tum_loc[0,0],
                                                                            w_seg2[0,0].round().long(),
                                                                            to_onehot=True, inter_dst=5), inter_dst=1).cpu()
                log_scalars['target_warped_ratio'] = ratio.mean().item()

                # calculate the vp loss based on change of TSR
                ## Notice!! It should be *ratio instead of /ratio
                k_sz = torch.zeros(1).cuda()
                det_flow = (jacobian_det(agg_flows[-1], return_det=True).abs()*ratio).clamp(min=1/3, max=3)
                vp_mask = F.interpolate(vp_tum_loc.float(), size=det_flow.shape[-3:], mode='trilinear', align_corners=False).float().squeeze() # B, S, H, W
                ## normalize by single voxel
                adet_flow = torch.where(det_flow>1, det_flow, (1/det_flow))
                if args.hyper_vp:
                    adet_flow = adet_flow*(hyp_param[:,0,None,None,None])
                k_sz_voxel = (adet_flow*vp_mask).sum()/vp_mask.sum()
                k_szs = (adet_flow*vp_mask).sum(dim=(-1,-2,-3)) / vp_mask.sum(dim=(-1,-2,-3))
                ## if normalize by each img in batch: check norm_img()

                # add vp_loss to loss
                k_sz_orig = k_sz + k_sz_voxel
                if not args.hyper_vp:
                    k_sz = k_sz_orig  * args.vol_preserve
                if torch.isnan(k_sz): k_sz = sim.new_zeros(1) # k_sz is all nan
                k_sz = reduce_mean(k_sz)
                loss = loss + k_sz

                with torch.no_grad():
                    vp_seg = F.interpolate(w_seg2, size=det_flow.shape[-3:], mode='trilinear', align_corners=False).float().squeeze().round().long() # B, S, H, W
                    img_dict['det_flow'] = visualize_3d(det_flow[0]).cpu()
                # log imgs
                img_dict['vp_det_flow'] = visualize_3d(adet_flow[0]).cpu()
                img_dict['vpdetflow_on_seg2'] = visualize_3d(draw_seg_on_vol(adet_flow[0], vp_seg[0], to_onehot=True, inter_dst=5), inter_dst=1).cpu()
                # log scalars
                log_scalars['k_sz_voxel'] = k_sz_voxel.item()
                loss_dict['vol_preserve_loss'] = k_sz_orig.item()

            if loss.isnan().any():
                # sometimes happen if data contatins no tumor, may need to check
                # import ipdb; ipdb.set_trace()
                loss = 0
            loss.backward()
            if args.hyper_vp and loss>0:
                # calculate loss and the gradient of the loss for each hyper_param
                total_norm = 0
                total_params = 0
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    total_params += p.numel()
                total_norm = total_norm ** (1. / 2)
                avg_norm = total_norm / len(hyp_param) / total_params
                # log them, keep 3 decimals
                log_list["hyp_value"] = hyp_param.tolist()
                log_list["sim_loss"] = sims.tolist()
                log_list["vp_loss"] = k_szs.tolist()
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

            # eval
            if iteration%10==0:
                avg_time_per_iter = (default_timer() - st_t) / (iteration + 1 - start_iter)
                est_time_for_epo = avg_time_per_iter * (len(train_loader) - iteration)
                est_time_for_train = (args.epochs - epoch - 1) * len(train_loader) * avg_time_per_iter + est_time_for_epo
                log_scalars.update({
                    'est_remain_time': est_time_for_train
                })
                if iteration<500 or iteration % 500 == 0:
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
                    if args.hyper_vp:
                        for k in log_list:
                            writer.add_histogram('train/'+k, np.array(log_list[k]), epoch * len(train_loader) + iteration)
                    if args.hyper_vp:
                        print('\n')
                        print("hyp param is  ", '\t'.join([str(i.item())[:5] for i in hyp_param]))
                        print("sim loss is:", "\t".join([str(i.item())[:5] for i in sims]))
                        print("vp loss is: ", "\t".join([str(i.item())[:5] for i in k_szs]))
                        print("avg hyper param:", hyp_param.mean().item(), "avg grad norm:", avg_norm, f"({total_params} params)")
                        # writer.add_scalar('train/avg_hyper_param', hyp_param.mean().item(), epoch * len(train_loader) + iteration)
                        # writer.add_scalar('train/avg_grad_norm', total_norm, epoch * len(train_loader) + iteration)

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
                                mask = seg2
                                if args.mask_seg_dice>0 and args.mask_seg_dice<1:
                                    mask = rand_mask_(mask)
                                moving_ = torch.cat([moving, mask], dim=1)
                            elif args.masked in ['soft', 'hard']:
                                input_seg, compute_mask = model.pre_register(fixed, moving, seg2, training=False, cfg=cfg_train)
                                moving_ = torch.cat([moving, input_seg], dim=1)
                            else:
                                moving_ = moving

                            if args.hyper_vp:
                                warped_, flows, agg_flows = model(fixed, moving_, hyp_input=fixed.new_zeros(fixed.shape[0], 1))
                            else: warped_, flows, agg_flows = model(fixed, moving_, )

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

            if iteration % ckp_freq == 0:
                if not args.debug and not os.path.exists('./ckp/model_wts/'+run_id):
                    os.makedirs('./ckp/model_wts/'+run_id)

                train_loss_log.append(train_epoch_loss / iteration)
                reg_loss_log.append(train_reg_loss / iteration)

                ckp = {}
                ckp['stem_state_dict'] = model.stems.state_dict()
                # save hypernet state dict if exists
                if args.hyper_vp:
                    ckp['hypernet_state_dict'] = model.hypernet.state_dict()

                ckp['train_loss'] = train_loss_log
                ckp['val_loss'] = val_loss_log
                ckp['epoch'] = epoch
                ckp['global_iter'] = iteration
                torch.save(ckp, f'{ckp_dir}/epoch_{epoch}.pth')
                optim_state = {}
                optim_state['optimizer_state_dict'] = optim.state_dict()
                optim_state['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(optim_state, f'{ckp_dir}/optim.pth')
            del fixed, moving, seg2, warped_, flows, agg_flows, loss, sim, reg,
            del loss_dict, log_scalars, data, img_dict
            t0 = default_timer()
        scheduler.step()
        start_iter = 0

if __name__ == '__main__':
    args = parser.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # set gpu to the one with most free memory
    import subprocess
    GPU_ID = subprocess.getoutput('nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader | nl -v 0 | sort -nrk 2 | cut -f 1| head -n 1 | xargs')
    print('Using GPU', GPU_ID)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    main(args)