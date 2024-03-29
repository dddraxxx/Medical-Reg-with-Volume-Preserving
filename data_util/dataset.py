from itertools import chain
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

    # keys method
    def keys(self):
        if self.file is None:
            return []
        return list(self.file.keys())

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


class RawData():
    def __init__(self, split_path, rounds=None, affine=None, scheme=None, **kwargs):
        with open(split_path, 'r') as f:
            config = json.load(f)
        self.files = FileManager(config['files'])
        self.subset = {}

        self.paired = {}
        for k, v in config['subsets'].items():
            self.subset[k] = {}
            # unravel v if v is a list of lists
            if len(v) and type(v[0]) == list:
                # remember the paired list
                self.paired[k] = v
                # unravel the list
                v = chain(*v)
            for entry in v:
                if entry.split('/')[-1] == '*':
                    entries = self.files.files[entry[:entry.rfind('/')]].keys()
                    for e in entries:
                        ee = entry.replace('*',e)
                        self.subset[k][ee] = self.files[ee]
                else:
                    self.subset[k][entry] = self.files[entry]

        for k, v in self.subset.items():
            print('Number of data in {} is {}'.format(k, len(v)))


        self.affine = affine
        self.config = config
        self.scheme = scheme
        if self.scheme is not None:
            def convert_int(key):
                try:
                    return int(key)
                except ValueError as e:
                    return key
            scheme = convert_int(scheme)
            self.schemes = dict([(convert_int(k), v)
                                for k, v in config['schemes'].items()])
            if self.affine:
                affine_dct = self.config.get('affine_matrix', {})
                assert len(self.schemes[scheme]) == 1
                self.affine_npy = [np.load(affine_dct[k], allow_pickle=True).item()
                                for k in self.schemes[scheme].items() if k in affine_dct][0]

            # no fraction used
            if 'lits_d'==scheme or 'lits_d' in self.schemes[scheme]:
                print('using lits_d')
                self.data_pairs = [(self.get_pairs_with_gt(self.subset[k]))
                            for k, fraction in self.schemes[scheme].items()]
            else:
                self.data_pairs = [(self.get_pairs(list(self.subset[k].values()), paired=self.paired.get(k, None)))
                            for k, fraction in self.schemes[scheme].items()]
            # chain the data pairs
            self.data_pairs = [item for sublist in self.data_pairs for item in sublist]
            self.rounds = rounds or len(self.data_pairs)

    def get_instance(self, id):
        return self.files[id]

    def get_pairs(self, data, unordered=True, paired=None):
        pairs = []
        if paired:
            # change the data to dict accoridng to the id
            dct_data = {d['id'].replace('_','/'): d for d in data}
            # return the data arranged by the paired list
            return [(dct_data[d1], dct_data[d2]) for d1, d2 in paired]
        for i, d1 in enumerate(data):
            for j, d2 in enumerate(data):
                if i != j:
                    # if ordered, only add the pairs with i < j; otherwise, add all pairs
                    if unordered or i < j:
                        pairs.append((d1, d2))
        return pairs

    def get_pairs_with_gt(self, dct):
        pairs = []
        for k in dct:
            if 'id2' not in dct[k]:
                not_added=True
                for v in dct:
                    if v != k and 'id2' in dct[v]:
                        for v2 in dct[v]['id2']:
                            if 'lits/{}'.format(v2) == k:
                                i2 = dict(dct[v]['id2'][v2])
                                i2['id'] = v2
                                pairs.append((dct[k], i2))
                                pairs.append((i2, dct[k]))
                                not_added=False
                                break
                    if not_added==False:
                        break
            else:
                for v in dct[k]['id2']:
                    i2 = dict(dct[k]['id2'][v])
                    i2['id'] = v
                    pairs.append((dct[k], i2))
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
            p1 = d1['point'][...]
            ret['point1'][:p1.shape[0],:] = p1
        if 'point' in d2:
            p2 = d2['point'][...]
            ret['point2'][:p2.shape[0],:] = p2
        # ret be np float
        for k in ['voxel1', 'voxel2', 'segmentation1', 'segmentation2', 'point1', 'point2']:
            ret[k] = ret[k][None].astype(np.float32)
        # normalize voxel
        ret['voxel1'] = ret['voxel1'] / 255.0
        ret['voxel2'] = ret['voxel2'] / 255.0
        if hasattr(self, 'precompute'):
            input_seg, compute_mask = self.precompute[ret['id2']]['input_seg'], self.precompute[ret['id2']]['compute_mask']
            ret['input_seg'] = input_seg[...]
            ret['compute_mask'] = compute_mask[...]
        return ret

class Data(RawData, Dataset):
    def __init__(self, args, **kwargs):
        RawData.__init__(self, args, **kwargs)

if __name__ == '__main__':
    data = Data('/home/hynx/regis/recursive_cascaded_networks/datasets/liver_cust.json', scheme='sliver')
    import sys
    sys.path.append('.')
    from tools.utils import show_img, combo_imgs
    import matplotlib.pyplot as plt

    print(len(data))
    n = 110
    for k in data[n]:
        if isinstance(data[0][k], np.ndarray):
            print(k, data[0][k].shape)
    img1 = data[n]['voxel1'][0]
    seg1 = data[n]['segmentation1'][0]
    point1 = data[n]['point1'][0]
    from PIL import Image, ImageDraw
    ims = []
    for p in point1:
        x, y, z = p
        im = show_img(img1[int(x)])
        draw = ImageDraw.Draw(im)
        draw.ellipse((y-5, z-5, y+5, z+5), fill='red')
        ims.append(np.array(im))
    print(point1)
    # combo_imgs(img1, seg1).save('img1.png')
    # combo_imgs(img1, *ims).save('img1.png')
