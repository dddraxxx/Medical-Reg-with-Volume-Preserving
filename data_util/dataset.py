import numpy as np
import json
import os
import h5py
import torch
from torch.utils.data import Dataset

class Split:
    TRAIN = 1
    VALID = 2

class Hdf5Reader:
    def __init__(self, path):
        try:
            self.file = h5py.File(path, "r")
        except Exception:
            print('{} not found!'.format(path))
            self.file = None

    def __getitem__(self, key):
        data = {'id': key}
        if self.file is None:
            return data
        group = self.file[key]
        for k in group:
            data[k] = group[k]
        return data


class FileManager:
    def __init__(self, files):
        self.files = {}
        for k, v in files.items():
            self.files[k] = Hdf5Reader(v["path"])

    def __getitem__(self, key):
        p = key.find('/')
        if key[:p] in self.files:
            ret = self.files[key[:p]][key[p+1:]]
            ret['id'] = key.replace('/', '_')
            return ret
        elif '/' in self.files:
            ret = self.files['/'][key]
            ret['id'] = key.replace('/', '_')
            return ret
        else:
            raise KeyError('{} not found'.format(key))


class Data(Dataset):
    def __init__(self, split_path, rounds=None, affine=None, paired=False, scheme=None):
        with open(split_path, 'r') as f:
            config = json.load(f)
        self.files = FileManager(config['files'])
        self.subset = {}

        for k, v in config['subsets'].items():
            self.subset[k] = {}
            for entry in v:
                self.subset[k][entry] = self.files[entry]

        self.paired = paired

        def convert_int(key):
            try:
                return int(key)
            except ValueError as e:
                return key
        self.schemes = dict([(convert_int(k), v)
                             for k, v in config['schemes'].items()])

        for k, v in self.subset.items():
            print('Number of data in {} is {}'.format(k, len(v)))

        self.affine = affine
        self.config = config
        self.scheme = scheme
        if self.affine:
            affine_dct = self.config.get('affine_matrix', {})
            assert len(self.schemes[scheme]) == 1
            self.affine_npy = [np.load(affine_dct[k], allow_pickle=True).item()
                            for k in self.schemes[scheme].items() if k in affine_dct][0]
        
        # no fraction used
        if 'lits_d'==scheme:
            print('using lits_d')
            self.data_pairs = [(self.get_pairs_with_gt(self.subset[k]))
                        for k, fraction in self.schemes[scheme].items()]
        else:
            self.data_pairs = [(self.get_pairs(list(self.subset[k].values())))
                        for k, fraction in self.schemes[scheme].items()]
        # chain the data pairs
        self.data_pairs = [item for sublist in self.data_pairs for item in sublist]
        self.rounds = rounds or len(self.data_pairs)

    def get_pairs(self, data, ordered=True):
        pairs = []
        for i, d1 in enumerate(data):
            for j, d2 in enumerate(data):
                if i != j:
                    if ordered or i < j:
                        pairs.append((d1, d2))
        return pairs
    
    def get_pairs_with_gt(self, dct):
        pairs = []
        for k in dct:
            for v in dct[k]['id2']:
                i2 = dict(dct[k]['id2'][v])
                i2['id'] = v
                pairs.append(dct[k], i2)
        return pairs
    
    def __len__(self):
        return self.rounds

    def __getitem__(self, idx):
        index = idx % len(self.data_pairs)
        d1, d2 = self.data_pairs[index]
        ret = {}
        ret['id1'] = d1['id']
        ret['id2'] = d2['id']
        if self.affine:
            id1 = d1['id']
            id2 = d2['id']
            total_matrix = self.affine_npy[id1][id2]
            ret['affine_matrix'] = total_matrix
        ret['voxel1'] = d1['volume'][...]
        ret['voxel2'] = d2['volume'][...]
        ret['segmentation1'] = np.zeros_like(ret['voxel1'])
        ret['segmentation2'] = np.zeros_like(ret['voxel2'])
        ret['point1'] = np.ones((6,3))*(-1)
        ret['point2'] = np.ones((6,3))*(-1)
        if 'segmentation' in d1:
            ret['segmentation1'] = d1['segmentation'][...]
        if 'segmentation' in d2:
            ret['segmentation2'] = d2['segmentation'][...]
        if 'point' in d1:
            ret['point1'] = d1['point'][...]
        if 'point' in d2:
            ret['point2'] = d2['point'][...]
        # ret be np float
        for k in ['voxel1', 'voxel2', 'segmentation1', 'segmentation2', 'point1', 'point2']:
            ret[k] = ret[k][None].astype(np.float32)
        # normalize voxel
        ret['voxel1'] = ret['voxel1'] / 255.0
        ret['voxel2'] = ret['voxel2'] / 255.0
        return ret

if __name__ == '__main__':
    data = Data('/home/hynx/regis/Recursive-Cascaded-Networks/datasets/liver_cust.json', scheme=1)
    print(len(data))
    for k in data[0]:
        if isinstance(data[0][k], np.ndarray):
            print(k, data[0][k].shape)

