from timeit import default_timer
import json
import argparse
import os
import time
import torch
from networks.recursive_cascade_networks import RecursiveCascadeNetwork
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from metrics.losses import score_metrics, total_loss
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import datetime as datetime
from torch.utils.tensorboard import SummaryWriter

# from data_util.ctscan import sample_generator
from data_util.dataset import Data, Split

parser = argparse.ArgumentParser()
parser.add_argument('-b', "--batch_size", type=int, default=4)
parser.add_argument('-n', "--n_cascades", type=int, default=5)
parser.add_argument('-e', "--epochs", type=int, default=5)
parser.add_argument("--round", type=int, default=20000)
parser.add_argument("-v", "--val_steps", type=int, default=1000)
parser.add_argument('-cf', "--checkpoint_frequency", type=int, default=20)
parser.add_argument('-c', "--checkpoint", type=str, default=None)
parser.add_argument('--fixed_sample', type=int, default=100)
parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('-d', '--dataset', type=str, default='datasets/liver_cust.json', help='Specifies a data config')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--debug', action='store_true', help="run the script without saving files")
parser.add_argument('--name', type=str, default='')

args = parser.parse_args()


def identify_axes(ax_dict, fontsize=48):
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)


def plot_grid(ax, flow, factor=10):
    """ Plot the grid generated by a flow. The displacement can be too small, so we add a scale factor"""
    grid = factor * flow[:, ::8, ::8]
    lin_range = np.linspace(0, 512, 64)
    x, y = np.meshgrid(lin_range, lin_range)
    x = x + grid[0, ...]
    y = y + grid[1, ...]
    y = y

    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, color='black', linewidths=0.8))
    ax.add_collection(LineCollection(segs2, color='black', linewidths=0.8))
    ax.autoscale()


def generate_plots(fixed, moving, warped, flows, train_loss, val_loss, reg_loss, epoch):
    """ Save some images and plots during training"""
    moving = moving.detach().cpu().numpy()
    fixed = fixed.detach().cpu().numpy()
    warped = [w.detach().cpu().numpy() for w in warped]
    flows = [f.detach().cpu().numpy() for f in flows]

    fig = plt.figure(constrained_layout=True, figsize=(4 * 5, 4 * 3))
    ax_dict = fig.subplot_mosaic("""
                                 FABCD
                                 LGHIE
                                 MKJWX
                                 """)

    ax_dict['F'].imshow(moving[0, 0, ...], cmap='gray')
    ax_dict['F'].set_title('Moving')

    ax_dict['W'].imshow(fixed[0, 0, ...], cmap='gray')
    ax_dict['W'].set_title('Fixed')

    for i, ax_name in enumerate(list("ABCDEX")):
        ax_dict[ax_name].imshow(warped[i][0, 0, ...], cmap='gray')
        if ax_name == "A":
            ax_dict[ax_name].set_title("Affine")
        else:
            ax_dict[ax_name].set_title(f"Cascade {i}")

    ax_dict['L'].plot(train_loss, color='red', label='train_loss')
    ax_dict['L'].plot(val_loss, label='val_loss', color='blue')
    ax_dict['L'].plot(reg_loss, label='train_reg_loss', color='green')
    ax_dict['L'].set_title("Losses")
    ax_dict['L'].grid()
    ax_dict['L'].set_xlim(0, args.e)
    ax_dict['L'].legend(loc='upper right')
    ax_dict['L'].scatter(len(train_loss) - 1, train_loss[-1], s=20, color='red')
    ax_dict['L'].scatter(len(val_loss) - 1, val_loss[-1], s=20, color='blue')
    ax_dict['L'].scatter(len(reg_loss) - 1, reg_loss[-1], s=20, color='green')

    for i, ax_name in enumerate(list("GHIJKM")):
        plot_grid(ax_dict[ax_name], flows[i][0, ...])
        if ax_name == "G":
            ax_dict[ax_name].set_title("Affine")
        else:
            ax_dict[ax_name].set_title(f"Cascade {i}")

    plt.suptitle(f"Epoch {epoch}")
    plt.savefig(f'./ckp/visualization/epoch_{epoch}.png')


def main():
    if not os.path.exists('./ckp/model_wts'):
        print("Creating ckp dir")
        os.makedirs('./ckp/model_wts')

    if not os.path.exists('./ckp/visualization'):
        print("Creating visualization dir")
        os.makedirs('./ckp/visualization')
    # Hong Kong time
    dt = datetime.datetime.now(tz=datetime.timezone(datetime.timedelta(hours=8)))
    run_id = dt.strftime('%b%d_%H%M%S')

    # mkdirs for log
    if not args.debug:
        log_dir = os.path.join('./logs', run_id)
        os.path.exists(log_dir) or os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)

    # read config
    with open(args.dataset, 'r') as f:
        cfg = json.load(f)
        image_size = cfg.get('image_size', [128, 128, 128])
        image_type = cfg.get('image_type')
        segmentation_class_value=cfg.get('segmentation_class_value', {'unknown':1})

    model = RecursiveCascadeNetwork(n_cascades=args.n_cascades, im_size=image_size)
    trainable_params = []
    for submodel in model.stems:
        trainable_params += list(submodel.parameters())

    trainable_params += list(model.reconstruction.parameters())

    lr = args.lr
    optim = Adam(trainable_params, lr=args.lr)
    scheduler = StepLR(optimizer=optim, step_size=10, gamma=0.96)
    print('train', Split.TRAIN)
    print('Val', Split.VALID)
    train_dataset = Data(args.dataset, rounds=args.round, scheme=Split.TRAIN)
    val_dataset = Data(args.dataset, scheme=Split.VALID)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Saving the losses
    train_loss_log = []
    reg_loss_log = []
    val_loss_log = []

    for epoch in range(1, args.epochs + 1):
        print(f"-----Epoch {epoch} / {args.epochs}-----")
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
            warped, flows = model(fixed, moving)
            loss, reg = total_loss(fixed, warped[-1], flows)
            loss.backward()
            optim.step()

            train_epoch_loss = train_epoch_loss + loss.item()
            train_reg_loss = train_reg_loss + reg.item()

            if iteration == args.fixed_sample:
                vis_batch.append(fixed)
                vis_batch.append(moving)
                vis_batch.append(warped)
                vis_batch.append(flows)

            if iteration%10==0 or args.debug:
                if iteration<500 or iteration % 500 == 0:
                    print('*%s* ' % run_id,
                          time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                          'Steps %d, Total time %.2f, data %.2f%%. Loss %.3e lr %.3e' % (iteration,
                                                                                         default_timer() - t0,
                                                                                         (t1 - t0) / (
                                                                                             default_timer() - t0),
                                                                                         loss,
                                                                                         lr),
                          end='\n')
                if not args.debug:
                    writer.add_scalar('train/loss', loss, epoch * len(train_loader) + iteration)
                    writer.add_scalar('train/reg', reg, epoch * len(train_loader) + iteration)
                    writer.add_scalar('train/lr', lr, epoch * len(train_loader) + iteration)

                if iteration%args.val_steps==0 or args.debug:
                    print(f">>>>> Validation <<<<<")
                    val_epoch_loss = 0
                    dice_loss = {k:0 for k in segmentation_class_value.keys()}
                    model.eval()
                    for iteration, data in enumerate(val_loader):
                        # if iteration % int(0.1 * len(val_loader)) == 0:
                        #     print(f"\t-----Iteration {iteration} / {len(val_loader)} -----")
                        with torch.no_grad():
                            fixed, moving = data['voxel1'], data['voxel2']
                            fixed = fixed.cuda()
                            moving = moving.cuda()
                            warped, flows = model(fixed, moving)
                            sim, reg = total_loss(fixed, warped[-1], flows)
                            loss = sim + reg
                            val_epoch_loss += loss.item()
                            # to do: add dice loss for each label
                            for k,v in segmentation_class_value.items():
                                seg1 = data['segmentation1'].cuda() > v-0.5
                                seg2 = data['segmentation2'].cuda() > v-0.5
                                w_seg2 = model.reconstruction(seg2.float(), flows[-1].float()) > 0.5
                                dice, jac = score_metrics(seg1, w_seg2)
                                dice_loss[k] += dice.mean().item()

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
            t0 = default_timer()

        train_loss_log.append(train_epoch_loss / len(train_loader))
        reg_loss_log.append(train_reg_loss / len(train_loader))

        scheduler.step()

        if epoch % args.checkpoint == 0:
            ckp = {}
            for i, submodel in enumerate(model.stems):
                ckp[f"cascade {i}"] = submodel.state_dict()

            ckp['train_loss'] = train_loss_log
            ckp['val_loss'] = val_loss_log
            ckp['epoch'] = epoch

            torch.save(ckp, f'./ckp/model_wts/epoch_{epoch}.pth')

        # generate_plots(vis_batch[0], vis_batch[1], vis_batch[2], vis_batch[3], train_loss_log, val_loss_log, reg_loss_log, epoch)

if __name__ == '__main__':
    main()