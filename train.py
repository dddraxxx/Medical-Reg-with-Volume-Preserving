from timeit import default_timer
import json
import argparse
import os
import time

import numpy as np
from tools.utils import load_model_from_dir, load_model
import torch
from torch.functional import F
from networks.recursive_cascade_networks import RecursiveCascadeNetwork
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, LambdaLR
from torch.utils.data import DataLoader
from metrics.losses import det_loss, jacobian_det, masked_sim_loss, ortho_loss, reg_loss, dice_jaccard, sim_loss, surf_loss
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
parser.add_argument("--round", type=int, default=20000)
parser.add_argument("-v", "--val_steps", type=int, default=1000)
parser.add_argument('-cf', "--checkpoint_frequency", default=0.5, type=float)
parser.add_argument('-c', "--checkpoint", type=str, default=None)
parser.add_argument('-ct', '--continue_training', action='store_true')
parser.add_argument('--ctt', '--continue_training_this', type=str, default=None)
parser.add_argument('--fixed_sample', type=int, default=100)
parser.add_argument('-g', '--gpu', type=str, default='', help='GPU to use')
parser.add_argument('-d', '--dataset', type=str, default='datasets/liver_cust.json', help='Specifies a data config')
parser.add_argument('-ts', '--training_scheme', type=str, default='', help='Specifies a training scheme')
parser.add_argument('-aug', '--augment', default=True, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='Augment data')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_scheduler', default = 'step', type=str, help='lr scheduler', choices=['linear', 'step', 'cosine'])
parser.add_argument('--debug', action='store_true', help="run the script without saving files")
parser.add_argument('--name', type=str, default='')
parser.add_argument('--ortho', type=float, default=0.1, help="use ortho loss")
parser.add_argument('--det', type=float, default=0.1, help="use det loss")
parser.add_argument('--reg', type=float, default=1, help="use reg loss")
parser.add_argument('-ks', '--keep_size', type=float, default=0, help="use keep-size loss")
parser.add_argument('-st', '--size_type', choices=['whole', 'organ', 'constant'], default='organ')
parser.add_argument('-m', '--masked', choices=['soft', 'seg', 'hard'], default='',
     help="mask the tumor part when calculating similarity loss")
parser.add_argument('-msk_thr', '--mask_threshold', type=float, default=2, help="volume changing threshold for mask")
parser.add_argument('-mn', '--masked_neighbor', type=int, default=5, help="for masked neibor calculation")
parser.add_argument('--stage1_rev', type=bool, default=False, help="whether to use reverse flow in stage 1")
parser.add_argument('-inv', '--invert_loss', action='store_true', help="invertibility loss")
parser.add_argument('--surf_loss', default=0, type=float, help='Surface loss weight')
parser.add_argument('-ic', '--in_channel', default=2, type=int, help='Input channel number')
parser.add_argument('-w_ksv', '--w_ks_voxel', default=1, type=float, help='Weight for voxel method in ks loss')
parser.add_argument('--ks_norm', default='voxel', choices=['image', 'voxel'])
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

    # mkdirs for log
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
        if args.ctt:
            print("Using log dir")
            log_dir = args.ctt
            ckp_dir = log_dir+'/model_wts/'
            writer = SummaryWriter(log_dir=log_dir)
        # record args
        with open(log_dir+'/args.txt', 'a') as f:
            from pprint import pprint
            pprint(args.__dict__, f)
    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    # read config
    with open(args.dataset, 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        image_type = cfg.get('image_type', None)
        segmentation_class_value=cfg.get('segmentation_class_value', {'unknown':1})

    in_channels = args.in_channel
    model = RecursiveCascadeNetwork(n_cascades=args.n_cascades, im_size=image_size, base_network=args.base_network, cr_aff=True, in_channels=in_channels).cuda()
    if args.masked == 'soft' or args.masked == 'hard':
        state_path = '/home/hynx/regis/recursive-cascaded-networks/logs/Jan08_180325_normal-vtn'
        print('Building stage 1 model from', state_path)
        stage1_model = RecursiveCascadeNetwork(n_cascades=args.n_cascades, im_size=image_size, base_network='VTN', cr_aff=True)
        load_model_from_dir(state_path, stage1_model)
        stage1_model = stage1_model.cuda().eval()
    # add checkpoint loading
    start_epoch = 0
    if args.checkpoint or args.ctt:
        ckp = args.checkpoint or ckp_dir
        if os.path.isdir(ckp):
            ckp = load_model_from_dir(ckp, model)
        else: load_model(torch.load(os.path.join(ckp)), model)
        print("Loaded checkpoint from {}".format(ckp))
        if args.continue_training or args.ctt:
            print("Continue training from checkpoint")
            start_epoch = torch.load(ckp)['epoch'] + 1
        
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        mmodel = model.module
    else: mmodel = model
    trainable_params = []
    for submodel in mmodel.stems:
        trainable_params += list(submodel.parameters())
    trainable_params += list(mmodel.reconstruction.parameters())

    lr = args.lr
    optim = Adam(trainable_params, lr=lr)
    if args.lr_scheduler == 'step':
        scheduler = StepLR(optimizer=optim, step_size=10, gamma=0.96)
    elif args.lr_scheduler == 'linear':
        min_lr = 1e-6
        scheduler = LambdaLR(optimizer=optim, lr_lambda=lambda epoch: min_lr + (lr - min_lr) * (1 - epoch / args.epochs), last_epoch= start_epoch-1)
    print('Train', Split.TRAIN)
    print('Val', Split.VALID)
    train_dataset = Data(args.dataset, rounds=args.round*args.batch_size, scheme=args.training_scheme or Split.TRAIN)
    val_dataset = Data(args.dataset, scheme=Split.VALID)

    num_worker = min(8, args.batch_size)
    if dist.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_worker, sampler=train_sampler)
    else:
        # train_sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=num_worker, shuffle=True)
    # val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=num_worker, shuffle=False)

    # Saving the losses
    train_loss_log = []
    reg_loss_log = []
    val_loss_log = []
    max_seg = max(segmentation_class_value.values())

    if args.masked in ['soft', 'hard']:
        template = list(train_dataset.subset['slits-temp'].values())[0]
        template_image, template_seg = template['volume'], template['segmentation']
        template_image = torch.tensor(np.array(template_image).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()/255.0
        template_seg = torch.tensor(np.array(template_seg).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()

    # use cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    for epoch in range(start_epoch, args.epochs):
        print(f"-----Epoch {epoch+1} / {args.epochs}-----")
        train_epoch_loss = 0
        train_reg_loss = 0
        vis_batch = []
        t0 = default_timer()
        for iteration, data in enumerate(train_loader):
            log_scalars = {}
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
            if args.augment:
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
                    _, _, s1_agg_flows = stage1_model(stage1_inputs, template_input, return_neg=args.stage1_rev)
                    s1_flow = s1_agg_flows[-1]
                    w_template_seg = stage1_model.reconstruction(template_seg.expand_as(template_input), s1_flow)
                    mask_fixing = w_template_seg[:fixed.shape[0]]
                    mask_moving = w_template_seg[fixed.shape[0]:]
                    # check whether s1_agg_flows should be reversed
                    s1_flow_moving = s1_flow[fixed.shape[0]:]
                    # find shrinking tumors
                    w_moving, _, rs1_agg_flows = stage1_model(template_input[:fixed.shape[0]], moving, return_neg=args.stage1_rev)
                    rs1_flow = rs1_agg_flows[-1]
                    # find the area ranked by dissimilarity
                    def dissimilarity(x, y):
                        x_mean = x-x.mean(dim=(1,2,3,4), keepdim=True)
                        y_mean = y-y.mean(dim=(1,2,3,4), keepdim=True)
                        correlation = x_mean*y_mean
                        return -correlation
                    dissimilarity_moving = dissimilarity(template_input[:fixed.shape[0]], w_moving[-1])
                flow_det_moving = jacobian_det(rs1_flow, return_det=True)
                flow_det_moving = flow_det_moving.unsqueeze(1).abs()
                # get n x n neighbors 
                n = args.masked_neighbor
                if args.stage1_rev: flow_ratio = 1/flow_det_moving
                else: flow_ratio = flow_det_moving
                flow_ratio = flow_ratio.clamp(1/4,4)
                sizes = F.interpolate(flow_ratio, size=fixed.shape[-3:], mode='trilinear', align_corners=False) \
                    if flow_ratio.shape[-3:]!=fixed.shape[-3:] else flow_ratio
                flow_ratio = F.avg_pool3d(sizes, kernel_size=n, stride=1, padding=n//2)
                flow_ratio[~(mask_moving>0.5)]=0
                # sizes = sizes/sizes.mean(dim=(2,3,4), keepdim=True)
                # log_scalars['zooming_q0.9'] = torch.quantile(sizes, .9)
                # log_scalars['zooming_q0.1'] = torch.quantile(sizes, .1)
                if args.masked=='soft':
                    # normalize soft mask
                    soft_mask = flow_ratio/flow_ratio.mean(dim=(2,3,4), keepdim=True)
                elif args.masked=='hard':
                        # thres = torch.quantile(w_n, .9)
                    thres = args.mask_threshold
                    hard_mask = flow_ratio > thres
                    with torch.no_grad():
                        warped_hard_mask = mmodel.reconstruction(hard_mask.float(), agg_flows[-1]) > 0.5
                    input_seg = hard_mask + mask_moving
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
                w_seg2 = mmodel.reconstruction(seg2.float(), agg_flows[-1])
                warped_tseg = w_seg2>1.5
            # add log scalar tumor/ratio
            if (seg2>1.5).sum().item() > 0:
                t_c = (warped_tseg.sum(dim=(1,2,3,4)).float() / (seg2 > 1.5).sum(dim=(1,2,3,4)).float())
                t_c = t_c[t_c.isnan()==False]
                log_scalars['tumor_change'] = torch.where(t_c>1, t_c, 1/t_c).mean().item() 

            # sim and reg loss
            if not args.masked:
                sim = sim_loss(fixed, warped[-1])
            else:
                mask = warped_tseg | (seg1==2)
                assert mask.requires_grad == False
                if args.masked == 'seg':
                    sim = masked_sim_loss(fixed, warped[-1], mask)
                    mask_fixing = seg1
                    mask_moving = seg2
                    kmask = w_seg2
                elif args.masked == 'soft':
                    # soft mask * sim loss per pixel
                    sim = masked_sim_loss(fixed, warped[-1], mask, soft_mask)
                    with torch.no_grad():
                        warped_mask = mmodel.reconstruction(soft_seg.float(), agg_flows[-1])
                    kmask = warped_mask
                elif args.masked =='hard':
                    # TODO: may consider mask outside organ regions
                    sim = masked_sim_loss(fixed, warped[-1], warped_hard_mask)
                    with torch.no_grad():
                        warped_mask = mmodel.reconstruction(input_seg.float(), agg_flows[-1])
                    kmask = warped_mask
            reg = reg_loss(flows[1:]) * args.reg

            # VP loss
            if args.keep_size!=0:
                if args.size_type=='whole':
                    kmask = kmask>0.5
                elif args.size_type=='organ': 
                    kmask = kmask>1.5
                bs = agg_flows[-1].shape[0]
                ratio = (mask_fixing>.5).view(bs,-1).sum(1)/(mask_moving>.5).view(bs, -1).sum(1); ratio=ratio.view(-1,1,1,1)
                k_sz = 0
                ### single voxel ratio
                if args.w_ks_voxel>0:
                    det_flow = (jacobian_det(agg_flows[-1], return_det=True).abs()/ratio).clamp(min=1/3, max=3)
                    kmask_rs = F.interpolate(kmask.float(), size=det_flow.shape[-3:], mode='trilinear', align_corners=False) > 0.5
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
                    kmask_rs = F.interpolate(kmask.float(), size=det_flow.shape[-3:], mode='trilinear', align_corners=False) > 0.5
                    det_flow[~kmask_rs[:,0]]=0
                    k_sz_volume = (det_flow.sum(dim=(-1,-2,-3))/kmask_rs[:,0].sum(dim=(-1,-2,-3))/ratio.squeeze()).clamp(1/3, 3)
                    k_sz_volume = torch.where(k_sz_volume>1, k_sz_volume, 1/k_sz_volume)
                    k_sz_volume = k_sz_volume[k_sz_volume.nonzero()].mean()
                    k_sz = k_sz + (1-args.w_ks_voxel)*k_sz_volume
                    log_scalars['k_sz_volume'] = k_sz_volume
                k_sz = k_sz  * args.keep_size
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
            else: ivt = sim.new_zeros(1)

            if args.surf_loss:
                surf = surf_loss(w_seg2>0.5, seg1>0.5, agg_flows[-1]) * args.surf_loss
            else: surf = sim.new_zeros(1)

            loss = sim+reg+det+ort+k_sz+ivt+surf
            if loss.isnan().any():
                import pdb; pdb.set_trace()
                loss = 0
            loss.backward()
            optim.step()
            # ddp reduce of loss
            loss, sim, reg = reduce_mean(loss), reduce_mean(sim), reduce_mean(reg)
            ort, det = reduce_mean(ort), reduce_mean(det)
            loss_dict = {
                'sim_loss': sim.item(),
                'reg_loss': reg.item(),
                'ortho_loss': ort.item(),
                'det_loss': det.item(),
                'keep_size_loss': k_sz.item(),
                'invert_loss': ivt.item(),
                'surf_loss': surf.item(),
                'loss': loss.item()
            }

            train_epoch_loss = train_epoch_loss + loss.item()
            train_reg_loss = train_reg_loss + reg.item()

            # if iteration == args.fixed_sample:
            #     vis_batch.append(fixed)
            #     vis_batch.append(moving)
            #     vis_batch.append(warped)
            #     vis_batch.append(flows)
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
                        writer.add_image('train/img1', fixed[0, :, 64], epoch * len(train_loader) + iteration)
                        writer.add_image('train/img2', moving[0, :, 64], epoch * len(train_loader) + iteration)
                        writer.add_image('train/warped', warped[-1][0, :, 64], epoch * len(train_loader) + iteration)
                        writer.add_image('train/seg1', seg1[0, :, 64]/max_seg, epoch * len(train_loader) + iteration)
                        writer.add_image('train/seg2', seg2[0, :, 64]/max_seg, epoch * len(train_loader) + iteration)

                        writer.add_image('train/affine_warpd', warped[0][0, :, 64], epoch * len(train_loader) + iteration)
                        writer.add_image('train/w_seg2', w_seg2[0, :, 64]/max_seg, epoch * len(train_loader) + iteration)
                        # add sizes and hard mask
                        if 'sizes' in locals() and 'hard_mask' in locals():
                            writer.add_image('train/grid_sizes', sizes[0, :, 64], epoch * len(train_loader) + iteration)
                            writer.add_image('train/hard_mask', hard_mask[0, :, 64], epoch * len(train_loader) + iteration)

                if not args.debug:
                    for k in loss_dict:
                        writer.add_scalar('train/'+k, loss_dict[k], epoch * len(train_loader) + iteration)
                    writer.add_scalar('train/lr', scheduler.get_last_lr()[0], epoch * len(train_loader) + iteration)
                    for k in log_scalars:
                        writer.add_scalar('train/'+k, log_scalars[k], epoch * len(train_loader) + iteration)

                if iteration%args.val_steps==0 or args.debug:
                    print(f">>>>> Validation <<<<<")
                    val_epoch_loss = 0
                    dice_loss = {k:0 for k in segmentation_class_value.keys()}
                    to_ratios = 0
                    total_to = 0
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
                            if args.masked:
                                moving_ = torch.cat([moving, seg2], dim=1)
                            else:
                                moving_ = moving
                            warped_, flows, agg_flows = mmodel(fixed, moving_)
                            warped = [i[:, :1] for i in warped_]
                            sim, reg = sim_loss(fixed, warped[-1]), reg_loss(flows[1:])
                            loss = sim + reg
                            val_epoch_loss += loss.item()
                            for k,v in segmentation_class_value.items():
                                seg1 = data['segmentation1'].cuda() > v-0.5
                                seg2 = data['segmentation2'].cuda() > v-0.5
                                w_seg2 = mmodel.reconstruction(seg2.float(), agg_flows[-1].float()) > 0.5
                                dice, jac = dice_jaccard(seg1, w_seg2)
                                dice_loss[k] += dice.mean().item()
                            # add metrics for to_ratio
                            w_seg2 = mmodel.reconstruction(seg2.float(), agg_flows[-1].float())
                            to_ratio = (torch.sum(w_seg2>1.5, dim=(2, 3, 4)) / torch.sum(seg1>1.5, dim=(2, 3, 4))) /(torch.sum(w_seg2>0.5, dim=(2, 3, 4)) / torch.sum(seg1>0.5, dim=(2, 3, 4)))
                            to_ratio = torch.where(to_ratio>1, 1/to_ratio, to_ratio)
                            to_ratio = to_ratio.nanmean().item()
                            # if not nan 
                            if not np.isnan(to_ratio):
                                to_ratios += to_ratio
                                total_to += 1

                    tb_imgs['img1'] = fixed
                    tb_imgs['img2'] = moving
                    tb_imgs['warped'] = warped[-1]
                    with torch.no_grad():
                        tb_imgs['seg1'] = data['segmentation1'].cuda()
                        tb_imgs['seg2'] = data['segmentation2'].cuda()
                        tb_imgs['w_seg2'] = mmodel.reconstruction(tb_imgs['seg2'], agg_flows[-1].float())

                    mean_val_loss = val_epoch_loss / len(val_loader)
                    print(f"Mean val loss: {mean_val_loss}")
                    val_loss_log.append(mean_val_loss)
                    mean_dice_loss = {}
                    for k, v in dice_loss.items():
                        mean_dc = v / len(val_loader)
                        mean_dice_loss[k] = mean_dc
                        print(f'Mean dice loss {k}: {mean_dc}')
                    mean_to_ratio = to_ratios / total_to
                    print(f'Mean to_ratio: {mean_to_ratio}')
                    if not args.debug:
                        writer.add_scalar('val/loss', mean_val_loss, epoch * len(train_loader) + iteration)
                        writer.add_scalar('val/to_ratio', mean_to_ratio, epoch * len(train_loader) + iteration)
                        for k in mean_dice_loss.keys():
                            writer.add_scalar(f'val/dice_{k}', mean_dice_loss[k], epoch * len(train_loader) + iteration)
                        # add img1, img2, seg1, seg2, warped, w_seg2
                        writer.add_image('val/img1', tb_imgs['img1'][0,:,64], epoch * len(train_loader) + iteration)
                        writer.add_image('val/img2', tb_imgs['img2'][0,:,64], epoch * len(train_loader) + iteration)
                        writer.add_image('val/seg1', tb_imgs['seg1'][0,:,64]/max_seg, epoch * len(train_loader) + iteration)
                        writer.add_image('val/seg2', tb_imgs['seg2'][0,:,64]/max_seg, epoch * len(train_loader) + iteration)
                        writer.add_image('val/warped', tb_imgs['warped'][0,:,64], epoch * len(train_loader) + iteration)
                        writer.add_image('val/w_seg2', tb_imgs['w_seg2'][0,:,64]/max_seg, epoch * len(train_loader) + iteration)

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

                torch.save(ckp, f'{ckp_dir}/epoch_{epoch}_iter_{iteration}.pth')

            t0 = default_timer()   
        scheduler.step()
        # generate_plots(vis_batch[0], vis_batch[1], vis_batch[2], vis_batch[3], train_loss_log, val_loss_log, reg_loss_log, epoch)

if __name__ == '__main__':
    main()