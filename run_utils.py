from pathlib import Path as pa
import torch
import numpy as np
from tools.utils import *

stage1_cfg = {
    'liver': {
        'VTN': './liver/VTN/1/Jan08_180325_normal-vtn',
        'VXM': './liver/VXM/1/Jan08_175614_normal-vxm',
    },
    'brain': {
        'VTN': './brain/VTN/1/Mar02-050938_1_VTNx3_normal___',
        'VXM': '',
    },
}
import os
for i in stage1_cfg:
    for j in stage1_cfg[i]:
        stage1_cfg[i][j] = os.path.realpath(stage1_cfg[i][j])

def build_precompute(model, dataset, cfg):
    template = list(dataset.subset['temp'].values())[0]
    template_image, template_seg = template['volume'], template['segmentation']
    template_image = torch.tensor(np.array(template_image).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()/255.0
    template_seg = torch.tensor(np.array(template_seg).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    state_path = stage1_cfg[cfg.data_type][cfg.base_network]
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
    import ml_collections
    cfg = ml_collections.ConfigDict(cfg)
    print(cfg)
    return cfg

# read_cfg('/home/hynx/regis/recursive-cascaded-networks/logs/liver/VTN/Feb15_013352_1-hard-vporgan.1/')
