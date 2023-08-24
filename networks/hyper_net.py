import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperModel(nn.Module):
    """
    HyperModel
    """
    def __init__(self, nb_hyp_params=1, nb_hyp_layers=6, nb_hyp_units=128):
        super().__init__()
        self.nb_hyp_params = nb_hyp_params
        self.nb_hyp_layers = nb_hyp_layers
        self.nb_hyp_units = nb_hyp_units

        self.hyper_layers = nn.ModuleList()
        last_shape = nb_hyp_params
        for i in range(self.nb_hyp_layers):
            self.hyper_layers.append(nn.Linear(last_shape, self.nb_hyp_units))
            last_shape = self.nb_hyp_units
    def forward(self, x):
        """
        Forward pass
        """
        for i in range(self.nb_hyp_layers):
            x = self.hyper_layers[i](x)
        return x

class HyperConv(nn.Module):
    """
    HyperConv: Apply to 2D or 3D conv
    """
    def __init__(self, rank, hyp_units, in_channels, out_channels, kernel_size, kernel_use_bias=True, bias_use_bias=True, stride=1, padding=1):
        super(HyperConv, self).__init__()
        self.rank = rank
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        if isinstance(kernel_size, int):
            self.kernel_size = [kernel_size] * rank
        prod_kernel_size = np.prod(self.kernel_size)

        self.hyperkernel = nn.Linear(hyp_units, prod_kernel_size * in_channels * out_channels, bias=kernel_use_bias)
        self.hyperbias = nn.Linear(hyp_units, out_channels, bias=bias_use_bias)

    def build_hyp(self, hyp_tensor):
        self.hyp_tensor = hyp_tensor

    def forward(self, x, hyp_tensor=None):
        """
        Forward pass
        """
        if hyp_tensor is None:
            hyp_tensor = self.hyp_tensor
        bs = x.shape[0]
        hyp_kernel = self.hyperkernel(hyp_tensor).view(bs, self.out_channels, self.in_channels, *self.kernel_size)
        hyp_bias = self.hyperbias(hyp_tensor).view(bs, self.out_channels)

        # Initialize an empty tensor to store the output
        output = []

        # Perform convolution for each sample in the batch
        conv_ = getattr(F, f'conv{self.rank}d')
        for i in range(bs):
            output.append(conv_(x[i].unsqueeze(0), hyp_kernel[i], hyp_bias[i], stride=self.stride, padding=self.padding))

        self.hyp_tensor=None

        return torch.cat(output, dim=0)