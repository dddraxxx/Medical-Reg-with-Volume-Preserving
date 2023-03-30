from pathlib import Path as pa
import torch
import numpy as np
from tools.utils import *

stage1_cfg = {
    'liver': {
        # 'VTN': './logs/liver/VTN/1/Jan08_180325_li1_VTN_normal-vtn',
        'VTN': '/home/hynx/regis/recursive-cascaded-networks/logs/liver/VTN/1/Jan08-180325_li1_VTN_normal-vtn',
        'VXM': './logs/liver/VXM/1/Mar01-191032_1_VXMx1_normal__',
        'TSM': './logs/liver/TSM/1/Mar02-033226_1_TSMx1_normal__'
    },
    'brain': {
        'VTN': './logs/brain/VTN/1/Mar02-050938_1_VTNx3_normal___',
        'VXM': './logs/brain/VXM/1/Mar02-141247_1_VXMx1_normal__',
        'TSM': './logs/brain/TSM/1/Mar02-202418_br1_TSMx1_normal__'
    },
}
import os
for i in stage1_cfg:
    for j in stage1_cfg[i]:
        stage1_cfg[i][j] = os.path.realpath(stage1_cfg[i][j])

def build_precompute(model, dataset, cfg):
    # get template image and segmentation
    template = list(dataset.subset['temp'].values())[0]
    template_image, template_seg = template['volume'], template['segmentation']
    # convert to torch tensors and move to GPU
    template_image = torch.tensor(np.array(template_image).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()/255.0
    template_seg = torch.tensor(np.array(template_seg).astype(np.float32)).unsqueeze(0).unsqueeze(0).cuda()
    # load pretrained model
    state_path = stage1_cfg[cfg.data_type][cfg.base_network]
    if cfg.stage1_rev:
        print('using rev_flow')
    return model.build_preregister(template_image, template_seg, state_path, cfg.base_network)

def read_cfg(model_path):
    # Traverse up the directory tree until we find the folder where the model is saved
    while 'model_wts' in model_path:
        model_path = pa(model_path).parent
    # Get the model name
    name = pa(model_path).stem.split('_')[-1]
    cfg_path = pa(model_path) / 'args.txt'
    # Read the first line of the file
    cfg = open(cfg_path).read().split('\n\n')[0]
    # Replace single quotes with double quotes
    cfg = cfg.replace("'", '"')
    # Convert the string to a dictionary
    cfg = eval(cfg)
    cfg['name'] = name
    # Convert the dictionary to a ConfigDict
    import ml_collections
    cfg = ml_collections.ConfigDict(cfg)
    print(cfg)
    return cfg

# read_cfg('/home/hynx/regis/recursive-cascaded-networks/logs/liver/VTN/Feb15_013352_1-hard-vporgan.1/')
