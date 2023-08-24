#  imoprt lib ------
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import ml_collections
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '.')))
from data_util.dataset import Data, Split, RawData


# TODO: build a dataloader that returns every thing we need for training ------
data_cfg = ml_collections.ConfigDict()
data_cfg.training_scheme = Split.TRAIN
data_cfg.dataset = 'datasets/liver_cust.json'
data_cfg.round = 20000
data_cfg.batch_size = 4
data_cfg.num_worker = 4


def get_data(cfg=data_cfg):
    # cd current dir
    prev_dir = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    cfgs = cfg
    train_scheme = cfgs.training_scheme or Split.TRAIN
    train_dataset = Data(cfgs.dataset, rounds=cfgs.round *
                         cfgs.batch_size, scheme=train_scheme)

    # cd back
    os.chdir(prev_dir)
    return train_dataset


def get_np_data(cfg=data_cfg):
    # cd current dir
    prev_dir = os.getcwd()
    os.chdir(os.path.dirname(__file__))

    cfgs = cfg
    train_scheme = cfgs.training_scheme or Split.TRAIN
    train_dataset = RawData(cfgs.dataset, rounds=cfgs.round *
                            cfg.batch_size, scheme=train_scheme)

    # cd back
    os.chdir(prev_dir)
    return train_dataset


def get_dataloader(data, cfg):
    train_loader = DataLoader(
        data, batch_size=cfg.batch_size, num_workers=cfg.num_worker, shuffle=True)
    return train_loader


class DataList(Dataset):
    def __init__(self, data, list_items):
        self.data = data
        self.list_items = list_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ret = []
        for k in self.list_items:
            ret.append(self.data[idx][k])
        return ret


def merge_cfg(cfg, data_cfg):
    merged_cfg = ml_collections.ConfigDict()
    merged_cfg.update(data_cfg)
    merged_cfg.update(cfg)
    cfg = merged_cfg
    return cfg


def get_loader(cfg=data_cfg):
    # merge with data_cfg
    cfg = merge_cfg(cfg, data_cfg)

    data = get_data(cfg)
    if cfg.return_list:
        list_items = cfg.return_list
        data = DataList(data, list_items)
    loader = get_dataloader(data, cfg)
    return loader


#%%
# configure model
