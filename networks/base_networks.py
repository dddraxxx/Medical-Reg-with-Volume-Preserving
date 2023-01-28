#%%
import torch
import torch.nn as nn
from torch.nn import ReLU, LeakyReLU
import torch.nn.functional as F
from torch.distributions.normal import Normal
from . import layers
import numpy as np

BASE_NETWORK = ['VTN', 'VXM']

def conv(dim=2):
    if dim == 2:
        return nn.Conv2d
    
    return nn.Conv3d


def trans_conv(dim=2):
    if dim == 2:
        return nn.ConvTranspose2d
    
    return nn.ConvTranspose3d


def convolve(in_channels, out_channels, kernel_size, stride, dim=2):
    return conv(dim=dim)(in_channels, out_channels, kernel_size, stride=stride, padding=1)

def convolveReLU(in_channels, out_channels, kernel_size, stride, dim=2):
    return nn.Sequential(ReLU, convolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def convolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=2, leakyr_slope=0.1):
    return nn.Sequential(LeakyReLU(leakyr_slope), convolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def upconvolve(in_channels, out_channels, kernel_size, stride, dim=2):
    return trans_conv(dim=dim)(in_channels, out_channels, kernel_size, stride, padding=1)


def upconvolveReLU(in_channels, out_channels, kernel_size, stride, dim=2):
    return nn.Sequential(ReLU, upconvolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def upconvolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=2):
    return nn.Sequential(LeakyReLU(0.1), upconvolve(in_channels, out_channels, kernel_size, stride, dim=dim))

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],
        [32, 32, 32, 32, 16, 16]
    ]
    return nb_features

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1, in_channels=2):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # configure encoder (down-sampling path)
        prev_nf = in_channels
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(convolveLeakyReLU(prev_nf, nf, dim=ndims, kernel_size=3, stride=2, leakyr_slope=0.1))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(convolveLeakyReLU(channels, nf, dim=ndims, kernel_size=3, stride=1, leakyr_slope=0.1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(convolveLeakyReLU(prev_nf, nf, dim=ndims, kernel_size=3, stride=1, leakyr_slope=0.1))
            prev_nf = nf
 
    def forward(self, x):

        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x

class VXM(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    # @store_config_args
    def __init__(self,
        im_size,
        flow_multiplier=1,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        int_steps=7,
        int_downsize=2,
        in_channels=2,
        bidir=False,
        use_probs=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(im_size)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        self.unet_model = Unet(
            im_size,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            in_channels=in_channels,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        self.flow.initialized = True
        setattr(self.flow, 'initialized', True)

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError('Flow variance has not been implemented in pytorch - set use_probs to False')

        # configure optional resize layers
        resize = int_steps > 0 and int_downsize > 1
        self.resize = layers.ResizeTransform(int_downsize, ndims) if resize else None
        self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims) if resize else None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in im_size]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.flow_multiplier = flow_multiplier

    def forward(self, source, target, return_preint=False, return_neg=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
        '''
        bidir = self.bidir or return_neg

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if bidir else None
        
        returns = [pos_flow]
        if bidir: returns.append(neg_flow)
        if return_preint: returns.append(preint_flow)
        returns = [r*self.flow_multiplier for r in returns]
        return returns if len(returns)>1 else returns[0]
        

class VTN(nn.Module):
    def __init__(self, im_size=(128,128,128), flow_multiplier=1., channels=16, in_channels=2):
        super(VTN, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.dim = dim = len(im_size)

        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1 = convolveLeakyReLU(in_channels, channels, 3, 2, dim=dim)
        self.conv2 = convolveLeakyReLU(channels, 2 * channels, 3, 2, dim=dim)
        self.conv3 = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2, dim=dim)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1, dim=dim)
        self.conv4 = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2, dim=dim)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1, dim=dim)
        self.conv5 = convolveLeakyReLU(8 * channels, 16 * channels, 3, 2, dim=dim)
        self.conv5_1 = convolveLeakyReLU(16 * channels, 16 * channels, 3, 1, dim=dim)
        self.conv6 = convolveLeakyReLU(16 * channels, 32 * channels, 3, 2, dim=dim)
        self.conv6_1 = convolveLeakyReLU(32 * channels, 32 * channels, 3, 1, dim=dim)

        self.pred6 = convolve(32 * channels, dim, 3, 1, dim=dim)
        self.upsamp6to5 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv5 = upconvolveLeakyReLU(32 * channels, 16 * channels, 4, 2, dim=dim)

        self.pred5 = convolve(32 * channels + dim, dim, 3, 1, dim=dim)  # 514 = 32 * channels + 1 + 1
        self.upsamp5to4 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv4 = upconvolveLeakyReLU(32 * channels + dim, 8 * channels, 4, 2, dim=dim)

        self.pred4 = convolve(16 * channels + dim, dim, 3, 1, dim=dim)  # 258 = 64 * channels + 1 + 1
        self.upsamp4to3 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv3 = upconvolveLeakyReLU(16 * channels + dim,  4 * channels, 4, 2, dim=dim)

        self.pred3 = convolve(8 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp3to2 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv2 = upconvolveLeakyReLU(8 * channels + dim, 2 * channels, 4, 2, dim=dim)

        self.pred2 = convolve(4 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp2to1 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv1 = upconvolveLeakyReLU(4 * channels + dim, channels, 4, 2, dim=dim)

        self.pred0 = upconvolve(2 * channels + dim, dim, 4, 2, dim=dim)

    def forward(self, fixed, moving, return_neg = False):
        concat_image = torch.cat((fixed, moving), dim=1)  # 2 x 512 x 512
        x1 = self.conv1(concat_image)  # 16 x 256 x 256
        x2 = self.conv2(x1)  # 32 x 128 x 128
        x3 = self.conv3(x2)  # 1 x 64 x 64 x 64
        x3_1 = self.conv3_1(x3)  # 64 x 64 x 64
        x4 = self.conv4(x3_1)  # 128 x 32 x 32
        x4_1 = self.conv4_1(x4)  # 128 x 32 x 32
        x5 = self.conv5(x4_1)  # 256 x 16 x 16
        x5_1 = self.conv5_1(x5)  # 256 x 16 x 16
        x6 = self.conv6(x5_1)  # 512 x 8 x 8
        x6_1 = self.conv6_1(x6)  # 512 x 8 x 8

        pred6 = self.pred6(x6_1)  # 2 x 8 x 8
        upsamp6to5 = self.upsamp6to5(pred6)  # 2 x 16 x 16
        deconv5 = self.deconv5(x6_1)  # 256 x 16 x 16
        concat5 = torch.cat([x5_1, deconv5, upsamp6to5], dim=1)  # 514 x 16 x 16

        pred5 = self.pred5(concat5)  # 2 x 16 x 16
        upsamp5to4 = self.upsamp5to4(pred5)  # 2 x 32 x 32
        deconv4 = self.deconv4(concat5)  # 2 x 32 x 32
        concat4 = torch.cat([x4_1, deconv4, upsamp5to4], dim=1)  # 258 x 32 x 32

        pred4 = self.pred4(concat4)  # 2 x 32 x 32
        upsamp4to3 = self.upsamp4to3(pred4)  # 2 x 64 x 64
        deconv3 = self.deconv3(concat4)  # 64 x 64 x 64
        concat3 = torch.cat([x3_1, deconv3, upsamp4to3], dim=1)  # 130 x 64 x 64

        pred3 = self.pred3(concat3)  # 2 x 63 x 64
        upsamp3to2 = self.upsamp3to2(pred3)  # 2 x 128 x 128
        deconv2 = self.deconv2(concat3)  # 32 x 128 x 128
        concat2 = torch.cat([x2, deconv2, upsamp3to2], dim=1)  # 66 x 128 x 128

        pred2 = self.pred2(concat2)  # 2 x 128 x 128
        upsamp2to1 = self.upsamp2to1(pred2)  # 2 x 256 x 256
        deconv1 = self.deconv1(concat2)  # 16 x 256 x 256
        concat1 = torch.cat([x1, deconv1, upsamp2to1], dim=1)  # 34 x 256 x 256

        pred0 = self.pred0(concat1)  # 2 x 512 x 512

        return pred0 * 20 * self.flow_multiplier  # why the 20?


class VTNAffineStem(nn.Module):
    def __init__(self, dim=1, channels=16, flow_multiplier=1., im_size=512, in_channels=2, flow_correct=True):
        super(VTNAffineStem, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.dim = dim

        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1 = convolveLeakyReLU(in_channels, channels, 3, 2, dim=self.dim)
        self.conv2 = convolveLeakyReLU(channels, 2 * channels, 3, 2, dim=dim)
        self.conv3 = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2, dim=dim)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1, dim=dim)
        self.conv4 = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2, dim=dim)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1, dim=dim)
        self.conv5 = convolveLeakyReLU(8 * channels, 16 * channels, 3, 2, dim=dim)
        self.conv5_1 = convolveLeakyReLU(16 * channels, 16 * channels, 3, 1, dim=dim)
        self.conv6 = convolveLeakyReLU(16 * channels, 32 * channels, 3, 2, dim=dim)
        self.conv6_1 = convolveLeakyReLU(32 * channels, 32 * channels, 3, 1, dim=dim)

        # I'm assuming that the image's shape is like (im_size, im_size, im_size)
        self.last_conv_size = im_size // (self.channels * 4)
        self.fc_loc = nn.Sequential(
            nn.Linear(512 * self.last_conv_size**dim, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 6*(dim - 1))
        )
        # self.fc_loc = nn.Sequential(
        #     nn.Linear(512 * channels * self.last_conv_size**dim, 6*(dim - 1)),
        # )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        """
        Identity Matrix
            | 1 0 0 0 |
        I = | 0 1 0 0 | 
            | 0 0 1 0 |
        """
        if dim == 3:
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))
        else:
            self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        self.create_flow = self.cr_flow if flow_correct else self.wr_flow
        
    def cr_flow(self, theta, size):
        shape = size[2:]
        flow = F.affine_grid(theta-torch.eye(len(shape), len(shape)+1, device=theta.device), size, align_corners=False)
        if len(shape) == 2:
            flow = flow[..., [1, 0]]
            flow = flow.permute(0, 3, 1, 2)
        elif len(shape) == 3:
            flow = flow[..., [2, 1, 0]]
            flow = flow.permute(0, 4, 1, 2, 3)
        flow = flow*flow.new_tensor(shape).view(-1, *[1 for _ in shape])/2
        return flow
    def wr_flow(self, theta, size):
        flow = F.affine_grid(theta, size, align_corners=False)  # batch x 512 x 512 x 2
        if self.dim == 2:
            flow = flow.permute(0, 3, 1, 2)  # batch x 2 x 512 x 512
        else:
            flow = flow.permute(0, 4, 1, 2, 3)
        return flow  
    def rev_affine(self, theta, dim=2):
        b = theta[:, :, dim:]
        inv_w = torch.inverse(theta[:, :dim, :dim])
        neg_affine = torch.cat([inv_w, -inv_w@b], dim=-1)
        return neg_affine
    def neg_flow(self, theta, size):
        neg_affine = self.rev_affine(theta, dim=self.dim)
        return self.create_flow(neg_affine, size)

    def forward(self, fixed, moving):
        concat_image = torch.cat((fixed, moving), dim=1)  # 2 x 512 x 512
        x1 = self.conv1(concat_image)  # 16 x 256 x 256
        x2 = self.conv2(x1)  # 32 x 128 x 128
        x3 = self.conv3(x2)  # 1 x 64 x 64 x 64
        x3_1 = self.conv3_1(x3)  # 64 x 64 x 64
        x4 = self.conv4(x3_1)  # 128 x 32 x 32
        x4_1 = self.conv4_1(x4)  # 128 x 32 x 32
        x5 = self.conv5(x4_1)  # 256 x 16 x 16
        x5_1 = self.conv5_1(x5)  # 256 x 16 x 16
        x6 = self.conv6(x5_1)  # 512 x 8 x 8
        x6_1 = self.conv6_1(x6)  # 512 x 8 x 8

        # Affine transformation
        xs = x6_1.view(-1, 512 * self.last_conv_size ** self.dim)
        if self.dim == 3:
            theta = self.fc_loc(xs).view(-1, 3, 4)
        else:
            theta = self.fc_loc(xs).view(-1, 2, 3)
        flow = self.create_flow(theta, moving.size())
        # theta: the affine param
        return flow, {'theta': theta}


if __name__ == "__main__":
    vtn_model = VTN()
    vtn_affine_model = VTNAffineStem(dim=3, im_size=16)
    torch.manual_seed(0)
    # theta = torch.rand(4, 3, 4)
    theta = torch.tensor([
    [2, 0.1, 0,0.2],
    [0.5, 2, 0,0.5],
    [0.2, 0, 2, 1],
], dtype=torch.float)[None]
    # theta = torch.randn(1,3,4)
    flow = vtn_affine_model.create_flow(theta, (1, 1, 16,16,16))
    neg_flow = vtn_affine_model.neg_flow(theta, (1, 1, 16,16,16))
    print(flow.shape)
    from layers import SpatialTransformer
    st = SpatialTransformer(flow.shape[2:])
    # agg_flow = neg_flow + st(flow, neg_flow)
    agg_flow = flow + torch.einsum('bij, bjxyz -> bixyz', theta[:,:3,:3], neg_flow)
    print(agg_flow.quantile(0.95),flow.quantile(0.95))

    # x = torch.randn(1, 1, 512, 512, 512)
    # y1 = vtn_model(x, x)
    # y2 = vtn_affine_model(x, x)
    # assert  y1.size() == y2.size()

    #%%
    from torchvision import transforms
    from PIL import Image
    import matplotlib.pyplot as plt

    # %matplotlib inline

    img_path = "/home/hynx/regis/recursive-cascaded-networks/tools/disp.png"
    img_torch = transforms.ToTensor()(Image.open(img_path))[:, :400,600:1000]

    plt.imshow(img_torch.numpy().transpose(1,2,0))
    plt.show()
    # %%
    from torch.nn import functional as F

    theta = torch.tensor([
        [2, 0, 0.2],
        [0, 2, 0.5]
    ], dtype=torch.float)
    grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size())
    output = F.grid_sample(img_torch.unsqueeze(0), grid)
    new_img_torch = output[0]
    plt.imshow(new_img_torch.numpy().transpose(1,2,0))
    plt.show()
    #%%
    idt = torch.eye(2,3)
    def rev_affine(theta, dim=2):
        b = theta[:,dim:]
        inv_w = torch.inverse(theta[:dim, :dim])
        neg_affine = torch.cat([inv_w, -inv_w@b], dim=-1)
        return neg_affine

    r_theta = rev_affine(theta)
    print(r_theta)

    st = SpatialTransformer(img_torch.shape[-2:])
    vtn_affine_model = VTNAffineStem(dim=2, im_size=400)

    flow = vtn_affine_model.create_flow(theta[None], (1,1,400,400))
    rev_flow = vtn_affine_model.create_flow(r_theta[None], (1,2,400,400))
    agg_flow = flow + torch.einsum('bij, bjxy -> bixy', theta[None][:,:2,:2], rev_flow)
    # agg_flow = rev_flow+st(flow, rev_flow)
    print(agg_flow.max(), flow.max())

    # grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size())
    # new_img = F.grid_sample(img_torch[None], grid)
    # plt.imshow(new_img[0].numpy().transpose(1,2,0))
    # plt.show()

    fl_img = st(img_torch[None], agg_flow)
    # fl_img = st(img_torch[None], (flow))
    # fl_img = st(fl_img, rev_flow)
    plt.imshow(fl_img[0].numpy().transpose(1,2,0))
    plt.show()