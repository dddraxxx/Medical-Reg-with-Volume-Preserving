from pathlib import Path as pa
import torch
import numpy as np
from tools.utils import *

def build_precompute(model, dataset, cfg):
    template = list(dataset.subset['temp'].values())[0]
    template_image, template_seg = template['volume'], template['segmentation']
    template_image = torch.tensor(np.array(template_image).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()/255.0
    template_seg = torch.tensor(np.array(template_seg).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    if cfg.data_type == 'liver':
        if cfg.base_network == 'VTN':
            state_path = '/home/hynx/regis/recursive-cascaded-networks/logs/liver/VTN/Jan08_180325_normal-vtn'
        elif cfg.base_network == 'VXM':
            # state_path = '/home/hynx/regis/recursive-cascaded-networks/logs/Jan08_180325_normal-vtn'
            state_path = '/home/hynx/regis/recursive-cascaded-networks/logs/liver/VXM/Jan08_175614_normal-vxm'
    elif cfg.data_type == 'brain':
        if cfg.base_network == 'VTN':
            # state_path = '/home/hynx/regis/recursive-cascaded-networks/logs/brain/VTN/Feb27_034554_br-normal'
            state_path = '/home/hynx/regis/recursive-cascaded-networks/logs/brain/VTN/mini/Mar01-165553_mini_VTNx3_br-normal'
        elif cfg.base_network == 'VXM':
            state_path = ''
    if cfg.stage1_rev:
        print('using rev_flow')
    model.build_preregister(template_image, template_seg, state_path, cfg.base_network)

def read_cfg(model_path):
    while 'model_wts' in model_path:
        model_path = pa(model_path).parent
    name = pa(model_path).stem.split('_')[-1]
    cfg_path = pa(model_path) / 'args.txt'
    # read till the first empty line
    cfg = open(cfg_path).read().split('\n\n')[0]
    # turn single quotes to double quotes
    cfg = cfg.replace("'", '"')
    cfg = eval(cfg)
    cfg['name'] = name
    print(cfg)
    return cfg

# read_cfg('/home/hynx/regis/recursive-cascaded-networks/logs/liver/VTN/Feb15_013352_1-hard-vporgan.1/')
