import torch
import torch.nn as nn
import torch.nn.functional as nnf

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size):
        super().__init__()

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow, mode='bilinear'):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=False, mode=mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()
        
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

import faiss
res = faiss.StandardGpuResources()  
def faiss_knn(reference_embeddings, test_embeddings, k):
    """
    Finds the k elements in reference_embeddings that are closest to each
    element of test_embeddings.
    Args:
        reference_embeddings: numpy array of size (num_samples, dimensionality).
        test_embeddings: numpy array of size (num_samples2, dimensionality).
        k: int, number of nearest neighbors to find
    """
    d = reference_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index)
    gpu_index_flat.add(reference_embeddings) #
    _, indices = index.search(test_embeddings, k)
    return indices

import frnn
class RevSpatialTransformer(nn.Module):
    """
    N-D RevSpatial Transformer
    """

    def __init__(self, size):
        super().__init__()
        self.size = size
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids, dim=-1)
        grid = grid.type(torch.FloatTensor).reshape(-1,3)
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, flow, k=1):
        assert flow.shape[2:]==self.size, 'flow shape should be (bs, 3, %d, %d, %d), found: %s' % (self.size[0], self.size[1], self.size[2], flow.shape)
        bs = flow.shape[0]
        flow = flow.permute(0, 2, 3, 4, 1).reshape(bs, -1, 3)
        points = self.grid + flow
        # repeat bs in dim 0
        grid = self.grid.repeat(bs, 1, 1)
        values = -flow
        dists, idxs, _, grid = frnn.frnn_grid_points(grid, points, K=k, r=self.size[0]/10, return_nn=False);
        # to do: add support for k>1
        rev_flow = frnn.frnn_gather(values, idxs)
        return rev_flow.reshape(bs, *self.size, 3).permute(0, 4, 1, 2, 3)
    
    # def _faiss(self, flow, k=1):
    #     assert flow.shape[2:]==self.size, 'flow shape should be (bs, 3, %d, %d, %d), found: %s' % (self.size[0], self.size[1], self.size[2], flow.shape)
    #     bs = flow.shape[0]
    #     flow = flow.permute(0, 2, 3, 4, 1).reshape(bs, -1, 3)
    #     points = self.grid + flow
    #     values = -flow
    #     # to do: add support for k>1
    #     for i in range(bs):
    #         knn_idx = faiss_knn(points[i], self.grid, k)
    #         vi = values[i][knn_idx]

    # def _sample_forward(self, grid, flow, mode='trilinear'):
    # from torch_cluster import nearest
    #     dim = len(flow.shape) - 2
    #     bs = flow.shape[0]
    #     grid = grid[None]
    #     spl_flow = nnf.interpolate(flow, size=self.size, mode='trilinear', align_corners=False)
    #     spl_grid = nnf.interpolate(grid, size=self.size, mode='trilinear', align_corners=False)
    #     pts = spl_flow + spl_grid
    #     rev_flow = flow.new_zeros(spl_flow.shape)
    #     # rev_flow = flow.new_zeros(flow.shape)
    #     q = spl_grid[0].moveaxis(0, -1).reshape(-1, dim)
    #     # q = grid[0].moveaxis(0, -1).reshape(-1, dim)
    #     for i in range(bs):
    #         p = pts[i].moveaxis(0, -1).reshape(-1, dim)
    #         idx = nearest(q, p)
    #         rev_flow[i] = (p[idx]-q[idx]).reshape(*flow.shape[2:], dim).moveaxis(-1, 0)
    #     rev_flow = nnf.interpolate(rev_flow, size=flow.shape[2:], mode='trilinear', align_corners=False)
    #     return -rev_flow
