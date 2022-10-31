from timeit import default_timer
import json
import argparse
import os
import time
from tools.utils import load_model_from_dir
import torch
from torch.functional import F
from networks.recursive_cascade_networks import RecursiveCascadeNetwork
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from metrics.losses import det_loss, jacobian_det, masked_sim_loss, ortho_loss, reg_loss, score_metrics, sim_loss
import datetime as datetime
from torch.utils.tensorboard import SummaryWriter
# from data_util.ctscan import sample_generator
from data_util.dataset import Data, Split
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser()
parser.add_argument('-b', "--batch_size", type=int, default=4)
parser.add_argument('-n', "--n_cascades", type=int, default=3)
parser.add_argument('-e', "--epochs", type=int, default=5)
parser.add_argument("--round", type=int, default=20000)
parser.add_argument("-v", "--val_steps", type=int, default=1000)
parser.add_argument('-cf', "--checkpoint_frequency", default=0.5, type=float)
parser.add_argument('-c', "--checkpoint", type=str, default=None)
parser.add_argument('--fixed_sample', type=int, default=100)
parser.add_argument('-g', '--gpu', type=str, default='', help='GPU to use')
parser.add_argument('-d', '--dataset', type=str, default='datasets/liver_cust.json', help='Specifies a data config')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--debug', action='store_true', help="run the script without saving files")
parser.add_argument('--name', type=str, default='')
parser.add_argument('--ortho', type=float, default=0.1, help="use ortho loss")
parser.add_argument('--keep_area', type=float, default=0, help="use keep-area loss")
parser.add_argument('-m', '--masked', choices=['soft', 'seg', 'hard'], default='',
     help="mask the tumor part when calculating similarity loss")
parser.add_argument('-mn', '--masked_neighbor', type=int, default=5, help="for masked neibor calculation")
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

    if not os.path.exists('./ckp/model_wts/'):
        print("Creating ckp dir")
        os.makedirs('./ckp/model_wts/')
    ckp_freq = int(args.checkpoint_frequency * args.round)

    if not os.path.exists('./ckp/visualization'):
        print("Creating visualization dir")
        os.makedirs('./ckp/visualization')

    # mkdirs for log
    if not args.debug:
        log_dir = os.path.join('./logs', run_id)
        os.path.exists(log_dir) or os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    # read config
    with open(args.dataset, 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        image_type = cfg.get('image_type', None)
        segmentation_class_value=cfg.get('segmentation_class_value', {'unknown':1})

    model = RecursiveCascadeNetwork(n_cascades=args.n_cascades, im_size=image_size).cuda()
    if args.masked == 'soft' or args.masked == 'hard':
        state_path = '/home/hynx/regis/recursive-cascaded-networks/ckp/model_wts/Sep27_145203_normal'
        stage1_model = RecursiveCascadeNetwork(n_cascades=args.n_cascades, im_size=image_size).cuda()
        load_model_from_dir(state_path, stage1_model)
    # add checkpoint loading
    if args.checkpoint:
        print("Loading checkpoint from {}".format(args.checkpoint))
        model.load_state_dict(torch.load(args.checkpoint))
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        mmodel = model.module
    else: mmodel = model
    trainable_params = []
    for submodel in mmodel.stems:
        trainable_params += list(submodel.parameters())

    trainable_params += list(mmodel.reconstruction.parameters())

    lr = args.lr
    optim = Adam(trainable_params, lr=args.lr)
    scheduler = StepLR(optimizer=optim, step_size=10, gamma=0.96)
    print('Train', Split.TRAIN)
    print('Val', Split.VALID)
    train_dataset = Data(args.dataset, rounds=args.round*args.batch_size, scheme=Split.TRAIN)
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
    for epoch in range(0, args.epochs):
        print(f"-----Epoch {epoch+1} / {args.epochs}-----")
        train_epoch_loss = 0
        train_reg_loss = 0
        vis_batch = []
        t0 = default_timer()
        for iteration, data in enumerate(train_loader):
            model.train()
            iteration += 1
            t1 = default_timer()
            optim.zero_grad()
            fixed, moving = data['voxel1'], data['voxel2']
            fixed = fixed.cuda()
            moving = moving.cuda()

            # do some augmentation
            seg1 = data['segmentation1'].cuda()
            moving, seg2 = model.augment(moving, data['segmentation2'].cuda())
            warped, flows, agg_flows, affine_params = model(fixed, moving, return_affine=True)

            # affine loss
            ortho_factor, det_factor = args.ortho, 0.1
            A = affine_params['theta'][..., :3, :3]
            ort = ortho_factor * ortho_loss(A)
            det = det_factor * det_loss(A)
            # sim and reg loss
            if not args.masked:
                sim = sim_loss(fixed, warped[-1])
            else:
                # caluculate mask
                t_seg2 = seg2==2
                # need torch no grad: Yes
                with torch.no_grad():
                    warped_tseg = mmodel.reconstruction(t_seg2.float(), agg_flows[-1])
                mask = (warped_tseg > 0.5) | (seg1==2)
                assert mask.requires_grad == False
                if args.masked == 'seg':
                    sim = masked_sim_loss(fixed, warped[-1], mask)
                elif args.masked == 'soft' or args.masked =='hard':
                    # get soft mask
                    with torch.no_grad():
                        _, _, s1_agg_flows = stage1_model(fixed, moving)
                    flow_det = jacobian_det(s1_agg_flows[-1].detach(), return_det=True)
                    flow_det = flow_det.unsqueeze(1).abs()
                    # get n x n neighbors 
                    n = args.masked_neighbor
                    flow_det = F.avg_pool3d(flow_det, kernel_size=n*2-1, stride=1, divisor_override=1)
                    # resize
                    areas = F.interpolate(flow_det, size=fixed.shape[-3:], mode='trilinear', align_corners=False)
                    if args.masked=='soft':
                        # normalize soft mask
                        w = torch.where(areas>1, 1/areas, areas)
                        soft_mask = w/w.mean(dim=(2,3,4), keepdim=True)
                        # soft mask * sim loss per pixel
                        sim = masked_sim_loss(fixed, warped[-1], mask, soft_mask)
                        # sim = masked_sim_loss(fixed, warped[-1], soft_mask.new_zeros(soft_mask.shape, dtype=bool), soft_mask)
                    elif args.masked=='hard':
                        areas[...,:n] = areas[...,-n:] = areas[...,:n,:] = areas[...,-n:,:] = areas[...,:n,:,:] = areas[...,-n:,:,:] = 0
                        thres = torch.quantile(areas, .95)
                        hard_mask = areas < thres
                        # hard mask
                        sim = masked_sim_loss(fixed, warped[-1], hard_mask)


            reg = reg_loss(flows[1:])
            if args.keep_area>0:
                kmask = seg2==2
                det_flow = jacobian_det(agg_flows[-1], return_det=True).abs().clamp(min=0.1, max=10)
                kmask = F.interpolate(kmask.float(), size=det_flow.shape[-3:], mode='trilinear', align_corners=False) > 0.5
                det_flow = det_flow[kmask[:,0]]
                area_loss = torch.where(det_flow>1, det_flow, 1/det_flow).mean() * args.keep_area
            else:
                area_loss = torch.zeros(1).cuda()
            loss = sim+reg+det+ort+area_loss
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
                'area_loss': area_loss.item(),
                'loss': loss.item()
            }

            train_epoch_loss = train_epoch_loss + loss.item()
            train_reg_loss = train_reg_loss + reg.item()

            # if iteration == args.fixed_sample:
            #     vis_batch.append(fixed)
            #     vis_batch.append(moving)
            #     vis_batch.append(warped)
            #     vis_batch.append(flows)
            if local_rank==0 and (iteration%10==0 or args.debug):
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
                        with torch.no_grad():
                            w_seg2 = mmodel.reconstruction(seg2, agg_flows[-1])
                        writer.add_image('train/w_seg2', w_seg2[0, :, 64]/max_seg, epoch * len(train_loader) + iteration)

                if not args.debug:
                    for k in loss_dict:
                        writer.add_scalar('train/'+k, loss_dict[k], epoch * len(train_loader) + iteration)
                    writer.add_scalar('train/lr', lr, epoch * len(train_loader) + iteration)

                if iteration%args.val_steps==0 or args.debug:
                    print(f">>>>> Validation <<<<<")
                    val_epoch_loss = 0
                    dice_loss = {k:0 for k in segmentation_class_value.keys()}
                    model.eval()
                    tb_imgs = {}
                    for itern, data in enumerate(val_loader):
                        itern += 1
                        if itern % int(0.33 * len(val_loader)) == 0:
                            print(f"\t-----Iteration {itern} / {len(val_loader)} -----")
                        with torch.no_grad():
                            fixed, moving = data['voxel1'], data['voxel2']
                            fixed = fixed.cuda()
                            moving = moving.cuda()
                            warped, flows, agg_flows = mmodel(fixed, moving)
                            sim, reg = sim_loss(fixed, warped[-1]), reg_loss(flows[1:])
                            loss = sim + reg
                            val_epoch_loss += loss.item()
                            for k,v in segmentation_class_value.items():
                                seg1 = data['segmentation1'].cuda() > v-0.5
                                seg2 = data['segmentation2'].cuda() > v-0.5
                                w_seg2 = mmodel.reconstruction(seg2.float(), agg_flows[-1].float()) > 0.5
                                dice, jac = score_metrics(seg1, w_seg2)
                                dice_loss[k] += dice.mean().item()
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
                    if not args.debug:
                        writer.add_scalar('val/loss', mean_val_loss, epoch * len(train_loader) + iteration)
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

                torch.save(ckp, f'./ckp/model_wts/{run_id}/epoch_{epoch}_iter_{iteration}.pth')

            t0 = default_timer()   
        scheduler.step()
        # generate_plots(vis_batch[0], vis_batch[1], vis_batch[2], vis_batch[3], train_loss_log, val_loss_log, reg_loss_log, epoch)

if __name__ == '__main__':
    main()