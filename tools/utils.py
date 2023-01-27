import torch
from torchvision.utils import make_grid, save_image, draw_segmentation_masks
import numpy as np
from PIL import Image

tt = torch.as_tensor

def interpolate_flow(flow, points):
    '''flow: n, c, d, h, w
    pionts: n, c, m'''
    n, c, d, h, w = flow.size()
    points = points.float()
    points_1 = points.floor() # n,c,m
    points_2 = points + points_1.new_tensor([1,0,0])[None,:,None]
    points_3 = points + points_1.new_tensor([0,1,0])[None,:,None]
    points_4 = points + points_1.new_tensor([0,0,1])[None,:,None]
    points_5 = points + points_1.new_tensor([1,1,0])[None,:,None]
    points_6 = points + points_1.new_tensor([1,0,1])[None,:,None]
    points_7 = points + points_1.new_tensor([0,1,1])[None,:,None]
    points_8 = points + points_1.new_tensor([1,1,1])[None,:,None]
    p1_flow = flow[:, :, points_1[:,0].long(), points_1[:,1].long(), points_1[:,2].long()]
    points_flow = flow[points_1] * (points_1 - points).abs().prod(dim=1, keepdim=True) + \
                    points_2 * (points_2 - points).abs().prod(dim=1, keepdim=True) + \
                    points_3 * (points_3 - points).abs().prod(dim=1, keepdim=True) + \
                    points_4 * (points_4 - points).abs().prod(dim=1, keepdim=True) + \
                    points_5 * (points_5 - points).abs().prod(dim=1, keepdim=True) + \
                    points_6 * (points_6 - points).abs().prod(dim=1, keepdim=True) + \
                    points_7 * (points_7 - points).abs().prod(dim=1, keepdim=True) + \
                    points_8 * (points_8 - points).abs().prod(dim=1, keepdim=True)
    return points_flow

def load_model(state_dict, model):
    # load state dict
    model.stems.load_state_dict(state_dict['stem_state_dict'])
    return model

def load_model_from_dir(checkpoint_dir, model):
    # glob file with suffix pth
    from pathlib import Path as pa
    import re
    p = pa(checkpoint_dir)
    # check if p has subdir named model_wts
    if (p/'model_wts').exists():
        p = p/'model_wts'
    p = p.glob('*.pth')
    p = sorted(p, key=lambda x: [int(n) for n in re.findall(r'\d+', str(x))])
    model_path = str(p[-1])
    load_model(torch.load(model_path), model)
    return model_path

import frnn
def get_nearest(ref, points, k, picked_points):
    dists, idxs, nn, grid = frnn.frnn_grid_points(points, ref, K=k, r=20, return_nn=False)
    nearest_points = frnn.frnn_gather(picked_points, idxs, k)
    return nearest_points

def visualize_3d(data, width=5, inter_dst=5, save_name=None, print_=False, color_channel: int=None, norm: bool=False):
    """
    data: (S, H, W) or (N, C, H, W)"""
    data =tt(data)
    if norm:
        data = norm(data)
    img = data.float()
    # st = torch.tensor([76, 212, 226])
    # end = st+128
    # img = img[st[0]:end[0],st[1]:end[1],st[2]:end[2]]
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
    if img.dim() < 4:
        img = img[:, None]
    img_s = img[::inter_dst]
    if color_channel is not None:
        img_s = img_s.movedim(color_channel, 1)

    img_f = make_grid(img_s, nrow=width, padding=5, pad_value=1, normalize=True)
    if save_name:
        save_image(img_f, save_name)
        if print_:
            print("Visualizing img with shape and type:", img_s.shape, img_s.dtype, "on path {}".format(save_name) if save_name else "")
        return range(0, img.shape[0], inter_dst)
    else:
        return img_f

def norm(data, dim=3, ct=False):
    data = tt(data)
    if ct:
        data1 = data.flatten(start_dim=-3)
        l = data1.shape[-1]
        upper = data.kthvalue(int(0.95*l), dim=-1)
        lower = data.kthvalue(int(0.05*l), dim=-1)
        data = data.clip(lower, upper)
    return PyTMinMaxScalerVectorized()(data, dim=3)

class PyTMinMaxScalerVectorized(object):
    """
    Transforms each channel to the range [0, 1].
    """

    def __call__(self, tensor: torch.Tensor, dim=2):
        """
        tensor: N*C*H*W"""
        tensor = tensor.clone()
        s = tensor.shape
        tensor = tensor.flatten(-dim)
        scale = 1.0 / (
            tensor.max(dim=-1, keepdim=True)[0] - tensor.min(dim=-1, keepdim=True)[0]
        )
        tensor.mul_(scale).sub_(tensor.min(dim=-1, keepdim=True)[0])
        return tensor.view(*s)

def draw_seg_on_vol(data, lb, if_norm=True, alpha=0.3, colors=["green", "red", "blue"]):
    """
    data: (1,) S, H, W
    label: (N,) S, H, W (binary value or bool)"""
    lb = tt(lb).float().reshape(-1, *lb.shape[-3:])
    data =tt(data).float().reshape(1, *data.shape[-3:])
    if if_norm:
        data = norm(data, 3)
    data = (data*255).cpu().to(torch.uint8)
    res = []
    for d, l in zip(data.transpose(0,1), lb.cpu().transpose(0,1)):
        res.append(draw_segmentation_masks(
                            (d).repeat(3, 1, 1),
                            l.bool(),
                            alpha=alpha,
                            colors=colors if len(l)<4 else None,
                        ))
    return torch.stack(res)/255
    
def show_img(res, save_path=None):
    import torchvision.transforms as T
    res = tt(res)
    if res.ndim>=3:
        return T.ToPILImage()(visualize_3d(res))
    # normalize res
    res = (res-res.min())/(res.max()-res.min())
    pimg = T.ToPILImage()(res)
    if save_path:
        pimg.save(save_path)
    return pimg

from PIL import Image
def combine_pil_img(*ims):
    widths, heights = zip(*(i.size for i in ims))
    total_width = sum(widths) + 5*(len(ims)-1)
    total_height = max(heights)
    im = Image.new('RGB', (total_width, total_height), color='white')
    for i, im_ in enumerate(ims):
        im.paste(im_, (sum(widths[:i])+5*i, 0))
    return im

combo_imgs = lambda *ims: combine_pil_img(*[show_img(i) for i in ims])

import matplotlib.pyplot as plt
def plt_img3d_show(imgs, cmap, intrv=5):
    fig, axes = plt.subplots((len(imgs+1))//(intrv**2), intrv)
    for ax, i in zip(axes.flatten(), range(0, len(imgs), intrv)):
        p = ax.imshow(imgs[i], cmap=cmap)
        # turnoff axis
        ax.axis('off')
        # title
        ax.set_title(f'{i}')
    plt.colorbar(p, ax=ax)
    # tight
    plt.tight_layout()
    plt.show()

def plt_img3d_axes(imgs, func, intrv=5, fig=None, figsize=(10, 10), **kwargs):
    if fig is None:
        fig, axes = plt.subplots((len(imgs+1))//(intrv**2), intrv, figsize=figsize)
    else:
        axes = fig.subplots((len(imgs+1))//(intrv**2), intrv)
    for ax, i in zip(axes.flatten(), range(0, len(imgs), intrv)):
        func(ax, i)
        # turnoff axis
        # ax.axis('off')
        # title
        ax.set_title(f'{i}')
    # tight
    plt.tight_layout()
    # plt.show()
    return axes, fig

import sys
sys.path.append(".")
from networks.layers import SpatialTransformer
import torch
def cal_single_warped(flow, img):
    if not isinstance(img, torch.Tensor):
        img = torch.tensor(img)
    if not isinstance(flow, torch.Tensor):
        flow = torch.tensor(flow)
    img = img.float()
    flow = flow.float()
    st = SpatialTransformer(img.shape)
    w_img = st(img[None,None], flow[None], mode='bilinear')
    return w_img[0,0]

from scipy.interpolate import griddata
def cal_rev_flow(flow):
    shape = flow.shape[:-1]
    xi = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32).reshape(3, -1).T
    points = xi+flow.reshape(-1, 3)
    values = -flow.reshape(-1,3)
    return griddata(points, values, xi, method='nearest').reshape(*flow.shape)