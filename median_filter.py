#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple, _ntuple
_sextuple = _ntuple(6, '_sextuple')

class MedianPool3d(nn.Module):
    """3D Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 3-tuple
         stride: pool stride, int or 3-tuple
         padding: pool padding, int or 6-tuple (d, h, w, d, h, w) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool3d, self).__init__()
        self.k = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _sextuple(padding)  # convert to d, h, w, d, h, w
        self.same = same

    def _padding(self, x):
        if self.same:
            id, ih, iw = x.size()[2:]
            if id % self.stride[0] == 0:
                pd = max(self.k[0] - self.stride[0], 0)
            else:
                pd = max(self.k[0] - (id % self.stride[0]), 0)
            if ih % self.stride[1] == 0:
                ph = max(self.k[1] - self.stride[1], 0)
            else:
                ph = max(self.k[1] - (ih % self.stride[1]), 0)
            if iw % self.stride[2] == 0:
                pw = max(self.k[2] - self.stride[2], 0)
            else:
                pw = max(self.k[2] - (iw % self.stride[2]), 0)
            pdl = pd // 2
            pdr = pd - pdl
            phl = ph // 2
            phr = ph - phl
            pwl = pw // 2
            pwr = pw - pwl
            padding = (pdl, pdr, phl, phr, pwl, pwr)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1]).unfold(4, self.k[2], self.stride[2])
        x = x.contiguous().view(x.size()[:5] + (-1,)).median(dim=-1)[0]
        return x

#%% test
if __name__ == '__main__':
    x = torch.randint(0, 10, (1, 1, 4, 4, 4), dtype=torch.float)
    print(x)
    print(MedianPool3d(kernel_size=3, stride=1, padding=0, same=True)(x))

