#%%
import argparse
import ml_collections

#%%
def parse_args(args = None):
    if args is None:
        parser = argparse.ArgumentParser()
    else:
        # args is dict
        args = {k.replace('--', ''):v for k,v in args.items()}
        # parse the given args instead of sys.argv
        parser = argparse.ArgumentParser(args)

    ## exp id/log  params
    parser.add_argument('--debug', action='store_true', help="run the script without saving files")
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('-wb', '--use_wandb', type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="whether to use wandb")

    ## training params
    parser.add_argument('-bs', "--batch_size", type=int, default=4)
    parser.add_argument('-base', '--base_network', type=str, default='VTN')
    parser.add_argument('-n', "--n_cascades", type=int, default=3)
    parser.add_argument('-e', "--epochs", type=int, default=5)
    parser.add_argument("-r", "--round", type=int, default=20000)
    parser.add_argument("-v", "--val_steps", type=int, default=1000)
    parser.add_argument('-cf', "--checkpoint_frequency", default=0.1, type=float)
    parser.add_argument('-ct', '--continue_training', action='store_true')
    parser.add_argument('--ctt', '--continue_training_this', type=lambda x: os.path.realpath(x), default=None)
    parser.add_argument('-g', '--gpu', type=str, default='', help='GPU to use')
    parser.add_argument('-aug', '--augment', default=False, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='Augment data')
    ## datasest params
    parser.add_argument('-d', '--dataset', type=str, default='datasets/liver_cust.json', help='Specifies a data config')
    parser.add_argument("-ts", "--training_scheme", type=lambda x:int(x) if x.isdigit() else x, default='', help='Specifies a training scheme')
    parser.add_argument('-vs', '--val_scheme', type=lambda x:int(x) if x.isdigit() else x, default='', help='Specifies a validation scheme')
    ## optimizer params
    parser.add_argument('--lr_scheduler', default = 'step', type=str, help='lr scheduler', choices=['linear', 'step', 'cosine'])
    parser.add_argument('--lr', type=float, default=1e-4)
    ## model params
    parser.add_argument('-ua', '--use_affine', type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="whether to use affine transformation")
    parser.add_argument('-c', "--checkpoint", type=lambda x: os.path.realpath(x), default=None)
    parser.add_argument('-ic', '--in_channel', default=2, type=int, help='Input channel number')

    ## loss params
    parser.add_argument('--ortho', type=float, default=0.1, help="use ortho loss")
    parser.add_argument('--det', type=float, default=0.1, help="use det loss")
    parser.add_argument('--reg', type=float, default=1, help="use reg loss")
    parser.add_argument('-inv', '--invert_loss', action='store_true', help="invertibility loss")
    parser.add_argument('--surf_loss', default=0, type=float, help='Surface loss weight')
    parser.add_argument('-dc', '--dice_loss', default=0, type=float, help='Dice loss weight')
    # for VP loss
    parser.add_argument('-vp', '--vol_preserve', type=float, default=0, help="use volume-preserving loss")
    parser.add_argument('-st', '--size_type', choices=['organ', 'tumor', 'tumor_gt', 'constant', 'dynamic', 'reg'], default='tumor', help = 'organ means VP works on whole organ, tumor means VP works on tumor region, tumor_gt means VP works on tumor region with ground truth, constant means VP ratio is a constant, dynamic means VP has dynamic weight, reg means VP is replaced by reg loss')
    parser.add_argument('--ks_norm', default='voxel', choices=['image', 'voxel'])
    parser.add_argument('-w_ksv', '--w_ks_voxel', default=1, type=float, help='Weight for voxel method in ks loss')


    # Stage 1 Params
    parser.add_argument('-m', '--masked', choices=['soft', 'seg', 'hard'], default='',
        help="mask the tumor part when calculating similarity loss")
    parser.add_argument('-msd', '--mask_seg_dice', type=float, default=-1, help="choose the accuracy of seg mask when using seg mask")
    parser.add_argument('-mt', '--mask_threshold', type=float, default=1.5, help="volume changing threshold for mask")
    parser.add_argument('-mn', '--masked_neighbor', type=int, default=3, help="for masked neibor calculation")
    parser.add_argument('-trsf', '--soft_transform', default='sigm', type=str, help='Soft transform')
    parser.add_argument('-bnd_thick', '--boundary_thickness', default=0.5, type=float, help='Boundary thickness')
    parser.add_argument('-use2', '--use_2nd_flow', default=False, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='Use 2nd flow')
    parser.add_argument('-useb', '--use_bilateral', default=True, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='Use bilateral filter')
    parser.add_argument('-pc', '--pre_calc', default=False, type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], help='Pre-calculate the flow')
    parser.add_argument('-s1r', '--stage1_rev', type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=False, help="whether to use reverse flow in stage 1")
    parser.add_argument('-os', '--only_shrink', type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="whether to only use shrinkage in stage 1")
    parser.add_argument('-uh', '--use_seg_help', type=lambda x:x.lower() in ['true', '1', 't', 'y', 'yes'], default=False, help="whether to use segmentation help in stage 1")

    # mask calculation needs an extra mask as input
    parser.set_defaults(in_channel=3 if parser.parse_args().masked else 2)
    parser.set_defaults(stage1_rev=True if parser.parse_args().base_network == 'VXM' else False)
    parser.set_defaults(n_cascades=1 if parser.parse_args().base_network != 'VTN' else 3)
    parser.set_defaults(use_affine=0 if parser.parse_args().base_network == 'DMR' else 1)

#%%
normal_network_cfg = ml_collections.ConfigDict()
normal_args = {
    'in_channel': 2,
}

#%%
vp_network_cfg = ml_collections.ConfigDict()