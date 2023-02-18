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
parser.add_argument('-msk_thr', '--mask_threshold', type=float, default=2, help="volume changing threshold for mask")
parser.add_argument('-mn', '--masked_neighbor', type=int, default=5, help="for masked neibor calculation")
parser.add_argument('-vp', '--vol_preserve', type=float, default=0, help="use volume-preserving loss")
parser.add_argument('-st', '--size_type', choices=['organ', 'tumor', 'tumor_gt', 'constant', 'dynamic', 'reg'], default='tumor', help = 'organ means VP works on whole organ, tumor means VP works on tumor region, tumor_gt means VP works on tumor region with ground truth, constant means VP ratio is a constant, dynamic means VP has dynamic weight, reg means VP is replaced by reg loss')
parser.add_argument('--ks_norm', default='voxel', choices=['image', 'voxel'])
parser.add_argument('-w_ksv', '--w_ks_voxel', default=1, type=float, help='Weight for voxel method in ks loss')
parser.add_argument('--stage1_rev', type=bool, default=False, help="whether to use reverse flow in stage 1")
parser.add_argument('-inv', '--invert_loss', action='store_true', help="invertibility loss")
parser.add_argument('--surf_loss', default=0, type=float, help='Surface loss weight')
parser.add_argument('-dc', '--dice_loss', default=0, type=float, help='Dice loss weight')
parser.add_argument('-ic', '--in_channel', default=2, type=int, help='Input channel number')
# mask calculation needs an extra mask as input
parser.set_defaults(in_channel=3 if parser.parse_args().masked else 2)


args = parser.parse_args()
if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

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
    # Hong Kong time
    dt = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
    run_id = dt.strftime('%b%d_%H%M%S') + (args.name and '_' + args.name)

    ckp_freq = int(args.checkpoint_frequency * args.round)

    # if not os.path.exists('./ckp/visualization'):
    #     print("Creating visualization dir")
    #     os.makedirs('./ckp/visualization')

    # mkdirs for log and Tensorboard
    if not args.debug:
        if not args.ctt:
            print("Creating log dir")
            log_dir = os.path.join('./logs', run_id)
            os.path.exists(log_dir) or os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)
            if not os.path.exists(log_dir+'/model_wts/'):
                print("Creating ckp dir")
                os.makedirs(log_dir+'/model_wts/')
                ckp_dir = log_dir+'/model_wts/'
        else:
            print("Using log dir")
            log_dir = args.ctt
            ckp_dir = log_dir+'/model_wts/'
            writer = SummaryWriter(log_dir=log_dir)
        # record args
        with open(log_dir+'/args.txt', 'a') as f:
            from pprint import pprint
            pprint(args.__dict__, f)
            command = "python train.py" + " ".join(["--{} {}".format(k, v) for k, v in args.__dict__.items()])
            print("\nRunning command:\n" + command, file=f)
    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    # read config
    with open(args.dataset, 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        image_type = cfg.get('image_type', None)
        segmentation_class_value=cfg.get('segmentation_class_value', {'unknown':1})

    train_scheme = args.training_scheme or Split.TRAIN
    val_scheme = Split.VALID
    print('Train', train_scheme)
    print('Val', val_scheme)
    train_dataset = Data(args.dataset, rounds=args.round*args.batch_size, scheme=train_scheme)
    val_dataset = Data(args.dataset, scheme=val_scheme)

    in_channels = args.in_channel
    model = RecursiveCascadeNetwork(n_cascades=args.n_cascades, im_size=image_size, base_network=args.base_network, in_channels=in_channels, compute_mask=False).cuda()
    if args.masked in ['soft', 'hard']:
        template = list(train_dataset.subset['slits-temp'].values())[0]
        template_image, template_seg = template['volume'], template['segmentation']
        template_image = torch.tensor(np.array(template_image).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()/255.0
        template_seg = torch.tensor(np.array(template_seg).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
        stage1_model = model.build_preregister(template_image, template_seg).RCN.cuda()

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
            start_epoch = torch.load(ckp)['epoch']
            start_iter = torch.load(ckp)['global_iter'] + 1
            print("Continue training from checkpoint from epoch {} iter {}".format(start_epoch, start_iter))
        
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
        for iteration, data in enumerate(train_loader, start=start_iter):
            # if start_iter is not 0, the 'if' statement will work
            if iteration>=len(train_loader): break
            log_scalars = {}
            loss_dict = {}
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
            if args.augment and (not args.debug):
                moving, seg2 = model.augment(moving, seg2)
            if args.invert_loss:
                fixed, moving = torch.cat([fixed, moving], dim=0), torch.cat([moving, fixed], dim=0)
                seg1, seg2 = torch.cat([seg1, seg2], dim=0), torch.cat([seg2, seg1], dim=0)
            
            # generate mask for tumor region according to flow
            if args.masked in ['soft', 'hard']:
                with torch.no_grad():
                    stage1_inputs = torch.cat([fixed, moving], dim=0)
                    template_input = template_image.expand_as(stage1_inputs)
                    # achieve organ mask via regristration
                    w_template, _, s1_agg_flows = stage1_model(stage1_inputs, template_input, return_neg=args.stage1_rev)
                    s1_flow = s1_agg_flows[-1]
                    w_template_seg = stage1_model.reconstruction(template_seg.expand_as(template_input), s1_flow)
                    mask_fixing = w_template_seg[:fixed.shape[0]]
                    mask_moving = w_template_seg[fixed.shape[0]:]
                    # find shrinking tumors
                    w_moving, _, rs1_agg_flows = stage1_model(template_input[:fixed.shape[0]], moving, return_neg=args.stage1_rev)
                    rs1_flow = rs1_agg_flows[-1]
                def filter_nonorgan(x, seg):
                    x[~(seg>0.5)] = 0
                    return x
                def extract_tumor_mask(stage1_moving_flow, moving_mask):
                    flow_det_moving = jacobian_det(stage1_moving_flow, return_det=True)
                    flow_det_moving = flow_det_moving.unsqueeze(1).abs()
                    # get n x n neighbors 
                    n = args.masked_neighbor
                    if args.stage1_rev: flow_ratio = 1/flow_det_moving
                    else: flow_ratio = flow_det_moving
                    flow_ratio = flow_ratio.clamp(1/3,3)
                    sizes = F.interpolate(flow_ratio, size=fixed.shape[-3:], mode='trilinear', align_corners=False) \
                        if flow_ratio.shape[-3:]!=fixed.shape[-3:] else flow_ratio
                    flow_ratio = F.avg_pool3d(sizes, kernel_size=n, stride=1, padding=n//2)
                    filter_nonorgan(flow_ratio, moving_mask)
                    globals().update(locals())
                    return flow_ratio
                flow_ratio = extract_tumor_mask(rs1_flow, mask_moving)
                if args.masked=='soft':
                    # normalize soft mask
                    soft_mask = flow_ratio/flow_ratio.mean(dim=(2,3,4), keepdim=True)
                elif args.masked=='hard':
                        # thres = torch.quantile(w_n, .9)
                    thres = args.mask_threshold
                    hard_mask = flow_ratio > thres
                    with torch.no_grad():
                        warped_hard_mask = mmodel.reconstruction(hard_mask.float(), rs1_agg_flows[-1]) > 0.5
                    input_seg = hard_mask + mask_moving
                if args.debug:
                    # visualize w_moving, seg2, moving
                    combo_imgs(*w_moving[-1][:,0]).save('w_moving.jpg')
                    with torch.no_grad():
                        # check the results of stage1 model on fixed-moving-registration
                        _, _, ss1ag_flows = stage1_model(fixed, moving)
                        wwseg2 = stage1_model.reconstruction(seg2, ss1ag_flows[-1])

                        w1_seg2 = stage1_model.reconstruction(seg2, rs1_flow)
                        w2_moving, _, rs2_agg_flows = stage1_model(template_input[:fixed.shape[0]], w_moving[-1])
                        w2_seg2 = stage1_model.reconstruction(w1_seg2, rs2_agg_flows[-1])
                        flow_ratio2 = extract_tumor_mask(rs2_agg_flows[-1], w2_seg2)
                        w3_moving, _, rs3_agg_flows = stage1_model(template_input[:fixed.shape[0]], w2_moving[-1])
                        w3_seg2 = stage1_model.reconstruction(w2_seg2, rs3_agg_flows[-1])
                        flow_ratio3 = extract_tumor_mask(rs3_agg_flows[-1], w3_seg2)
                        # find the area ranked by dissimilarity
                    def dissimilarity(x, y):
                        x_mean = x-x.mean(dim=(1,2,3,4), keepdim=True)
                        y_mean = y-y.mean(dim=(1,2,3,4), keepdim=True)
                        correlation = x_mean*y_mean
                        return -correlation
                    def dissimilarity_mask(x, y, mask):
                        x_mean = (x*mask).sum(dim=(1,2,3,4), keepdim=True)/mask.sum(dim=(1,2,3,4), keepdim=True)
                        y_mean = (y*mask).sum(dim=(1,2,3,4), keepdim=True)/mask.sum(dim=(1,2,3,4), keepdim=True)
                        x_ = x-x_mean
                        y_ = y-y_mean
                        correlation = x_*y_
                        correlation[~mask] = correlation[mask].median()
                        return -correlation
                    # dissimilarity_moving = dissimilarity_mask(template_input[:fixed.shape[0]], w_moving[-1], w1_seg2>.5)
                    # dissimilarity_moving2 = dissimilarity_mask(template_input[:fixed.shape[0]], w2_moving[-1], w2_seg2>.5)
                    # combo_imgs(*dissimilarity_moving[:,0]).save('dissimilarity_moving.jpg')
                    # combo_imgs(*dissimilarity_moving2[:, 0]).save("dissimilarity_moving2.jpg")
                    # what if we use the second stage flow?
                    combo_imgs(*w2_moving[-1][:,0]).save('w2_moving.jpg')
                    combo_imgs(*flow_ratio2[:,0]).save('flow_ratio2.jpg')
                    # or the third stage flow?
                    combo_imgs(*w3_moving[-1][:,0]).save('w3_moving.jpg')
                    combo_imgs(*flow_ratio3[:,0]).save('flow_ratio3.jpg')

                    combo_imgs(*w_template[-1][[1,2,5,6],0]).save('w_template.jpg')
                    combo_imgs(*w_template_seg[:,0]).save('w_template_seg.jpg')
                    combo_imgs(*wwseg2[:,0]).save('wwseg2.jpg')
                    combo_imgs(*mask_fixing[:,0]).save('mask_fixing.jpg')
                    combo_imgs(*mask_moving[:,0]).save('mask_moving.jpg')
                    combo_imgs(*moving[:,0]).save('moving.jpg')
                    combo_imgs(*seg2[:,0]).save('seg2.jpg')
                    combo_imgs(*flow_ratio[:,0]).save('flow_ratio.jpg')
                    combo_imgs(*hard_mask[:,0]).save('hard_mask.jpg')
                    # import debugpy; debugpy.listen(5678); print('Waiting for debugger attach'); debugpy.wait_for_client(); debugpy.breakpoint()
            elif args.masked=='seg': input_seg = seg2.float()
            
            if args.in_channel==3:
                moving_ = torch.cat([moving, input_seg], dim=1)
            else:
                moving_ = moving
            warped_, flows, agg_flows, affine_params = model(fixed, moving_, return_affine=True)
            warped = [i[:, :-1] for i in warped_] if args.in_channel>2 else warped_

            # affine loss
            ortho_factor, det_factor = args.ortho, args.det
            A = affine_params['theta'][..., :3, :3]
            ort = ortho_factor * ortho_loss(A)
            det = det_factor * det_loss(A)
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
                log_scalars['liver_change'] = o_c_nn.mean().item()
                log_scalars['to_ratio'] = torch.where(t_c_nn/o_c_nn >1, t_c_nn/o_c_nn, o_c_nn/t_c_nn).mean().item()

            # add log scalar dice_liver
            with torch.no_grad():
                dice_liver, _ = dice_jaccard(w_seg2>0.5, seg1>0.5)
                log_scalars['dice_liver'] = dice_liver.mean().item()

            # sim and reg loss
            # default masks used for VP loss
            mask = warped_tseg | (seg1>1.5)
            mask_fixing = seg1
            mask_moving = seg2
            vp_seg_mask = w_seg2
            if not args.masked:
                sim = sim_loss(fixed, warped[-1])
            else:
                assert mask.requires_grad == False
                if args.masked == 'seg':
                    sim = masked_sim_loss(fixed, warped[-1], mask)
                elif args.masked == 'soft':
                    # soft mask * sim loss per pixel
                    sim = masked_sim_loss(fixed, warped[-1], mask, soft_mask)
                    with torch.no_grad():
                        warped_mask = mmodel.reconstruction(soft_seg.float(), agg_flows[-1])
                    vp_seg_mask = warped_mask
                elif args.masked =='hard':
                    # TODO: may consider mask outside organ regions
                    sim = masked_sim_loss(fixed, warped[-1], warped_hard_mask)
                    with torch.no_grad():
                        warped_mask = mmodel.reconstruction(input_seg.float(), agg_flows[-1])
                    vp_seg_mask = warped_mask
            reg = reg_loss(flows[1:]) * args.reg

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
            if args.vol_preserve>0 and args.size_type=='reg':
                vp_loc = vp_seg_mask>0.5
                def reg_mask(flow, mask):
                    deno = mask.sum(dim=(1,2,3,4)).float()
                    dy = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
                    dx = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
                    dz = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
                    dy = dy * mask[:, :, 1:, :, :]
                    dx = dx * mask[:, :, :, 1:, :]
                    dz = dz * mask[:, :, :, :, 1:]
                    d = (dy**2).sum(dim=(1,2,3,4))/deno + (dx**2).sum(dim=(1,2,3,4))/deno + (dz**2).sum(dim=(1,2,3,4))/deno
                    return d.mean()/2.0 *4.0

                k_sz = reg_mask(agg_flows[-1], vp_loc) * args.vol_preserve
            elif args.vol_preserve>0:
                # to decide the voxels to be preserved
                if args.size_type=='organ':
                    # keep the volume of the whole organ 
                    vp_loc = vp_seg_mask>0.5
                elif args.size_type=='tumor': 
                    # keep the volume of the tumor region
                    vp_loc = vp_seg_mask>1.5
                elif args.size_type=='tumor_gt':
                    # use warped gt mask to know exactly the location of tumors
                    vp_loc = w_seg2>1.5
                    vp_seg_mask = w_seg2
                # TODO: add dynamic VP loss
                elif args.size_type=='dynamic':
                    # not feasible, flow_ratio contradicts volume preservation
                    w_flow_ratio = mmodel.reconstruction(flow_ratio, model.composite_flow(rs1_agg_flows[-1], agg_flows[-1]))
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
                    kmask_rs = F.interpolate(vp_loc.float(), size=det_flow.shape[-3:], mode='trilinear', align_corners=False) > 0.5
                    if args.ks_norm=='voxel':
                        det_flow = det_flow[kmask_rs[:,0]]
                        k_sz_voxel = torch.where(det_flow>1, det_flow, 1/det_flow).mean()
                    elif args.ks_norm=='image':
                        # normalize loss for every image
                        k_sz_voxel = (torch.where(det_flow>1, det_flow, 1/det_flow)*(kmask_rs[:,0])).sum(dim=(-1,-2,-3))
                        nonzero_idx = k_sz_voxel.nonzero()
                        k_sz_voxel = k_sz_voxel[nonzero_idx]/kmask_rs[:,0][nonzero_idx].sum(dim=(-1,-2,-3))
                        k_sz_voxel = k_sz_voxel.mean()
                    else: raise NotImplementedError
                    k_sz = k_sz + args.w_ks_voxel*k_sz_voxel
                    log_scalars['k_sz_voxel'] = k_sz_voxel
                ### whole volume ratio
                if args.w_ks_voxel<1:
                    det_flow = jacobian_det(agg_flows[-1], return_det=True).abs()
                    kmask_rs = F.interpolate(vp_loc.float(), size=det_flow.shape[-3:], mode='trilinear', align_corners=False) > 0.5
                    det_flow[~kmask_rs[:,0]]=0
                    k_sz_volume = (det_flow.sum(dim=(-1,-2,-3))/kmask_rs[:,0].sum(dim=(-1,-2,-3))/ratio.squeeze()).clamp(1/3, 3)
                    k_sz_volume = torch.where(k_sz_volume>1, k_sz_volume, 1/k_sz_volume)
                    k_sz_volume = k_sz_volume[k_sz_volume.nonzero()].mean()
                    k_sz = k_sz + (1-args.w_ks_voxel)*k_sz_volume
                    log_scalars['k_sz_volume'] = k_sz_volume
                k_sz = k_sz  * args.vol_preserve
                if torch.isnan(k_sz): k_sz = sim.new_zeros(1) # k_sz is all nan
            else:   
                k_sz = sim.new_zeros(1)
            
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

            loss = loss+sim+reg+det+ort+k_sz
            if loss.isnan().any():
                import pdb; pdb.set_trace()
                loss = 0
            loss.backward()
            optim.step()
            # ddp reduce of loss
            loss, sim, reg = reduce_mean(loss), reduce_mean(sim), reduce_mean(reg)
            ort, det = reduce_mean(ort), reduce_mean(det)
            loss_dict.update({
                'sim_loss': sim.item(),
                'reg_loss': reg.item(),
                'ortho_loss': ort.item(),
                'det_loss': det.item(),
                'vol_preserve_loss': k_sz.item(),
                'loss': loss.item()
            })

            train_epoch_loss = train_epoch_loss + loss.item()
            train_reg_loss = train_reg_loss + reg.item()

            if local_rank==0 and (iteration%10==0):
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
                    if not args.debug:
                        writer.add_image('train/img1', visualize_3d(fixed[0, 0]), epoch * len(train_loader) + iteration)
                        writer.add_image('train/img2', visualize_3d(moving[0, 0]), epoch * len(train_loader) + iteration)
                        writer.add_image('train/warped', visualize_3d(warped[-1][0, 0]), epoch * len(train_loader) + iteration)
                        writer.add_image('train/seg1', visualize_3d(seg1[0, 0]), epoch * len(train_loader) + iteration)
                        writer.add_image('train/seg2', visualize_3d(seg2[0, 0]), epoch * len(train_loader) + iteration)
                        writer.add_image('train/w_seg2', visualize_3d(w_seg2[0, 0]), epoch * len(train_loader) + iteration)
                        # add sizes and hard mask
                        if 'hard_mask' in locals():
                            tum_on_wimg = draw_seg_on_vol(moving[0,0], hard_mask[0,0])
                            org_on_wimg = draw_seg_on_vol(moving[0,0], mask_moving[0,0])
                            writer.add_image('train/hard_tum_mask', visualize_3d(tum_on_wimg), epoch * len(train_loader) + iteration)
                            writer.add_image('train/hard_org_mask', visualize_3d(org_on_wimg), epoch * len(train_loader) + iteration)

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
                        if itern % int(0.33 * len(val_loader)) == 0:
                            print(f"\t-----Iteration {itern} / {len(val_loader)} -----")
                        with torch.no_grad():
                            fixed, moving = data['voxel1'], data['voxel2']
                            seg2 = data['segmentation2'].cuda()
                            fixed = fixed.cuda()
                            moving = moving.cuda()
                            if args.masked=='seg':
                                moving_ = torch.cat([moving, seg2], dim=1)
                            else:
                                moving_ = moving
                            warped_, flows, agg_flows = mmodel(fixed, moving_)
                            warped = [i[:, :1] for i in warped_]
                            # you cannot use computed masked here because w_seg2 is used for evaluation and needs to be GT
                            # if args.masked:
                                # w_seg2 = warped_[-1][:, 1:]
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

                torch.save(ckp, f'{ckp_dir}/epoch_{epoch}_iter_{iteration}.pth')

            t0 = default_timer()   
        scheduler.step()
        start_iter = 0

if __name__ == '__main__':
    main()