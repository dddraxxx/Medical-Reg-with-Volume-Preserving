import torch
from torchvision.utils import make_grid, save_image, draw_segmentation_masks
import numpy as np
from PIL import Image

tt = torch.as_tensor

def visualize_3d(data, width=5, inter_dst=5, save_name=None):
    """
    data: (S, H, W) or (N, C, H, W)"""
    data =tt(data)
    img = data.float()
    # st = torch.tensor([76, 212, 226])
    # end = st+128
    # img = img[st[0]:end[0],st[1]:end[1],st[2]:end[2]]
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img).float()
    if img.dim() < 4:
        img = img[:, None]
    img = img[::inter_dst]
    print("Visualizing img with shape and type:", img.shape, img.dtype, "on path {}".format(save_name) if save_name else "")

    img_f = make_grid(img, nrow=width, padding=5, pad_value=1, normalize=True)
    if save_name:
        save_image(img_f, save_name)
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
    data: 1, S, H, W
    label: N, S, H, W (binary value or bool)"""
    lb = tt(lb).float()
    data =tt(data).float()
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
    
def show_img(res):
    import torchvision.transforms as T
    res = tt(res)
    if res.ndim>=3:
        return T.ToPILImage()(visualize_3d(res))
    # normalize res
    res = (res-res.min())/(res.max()-res.min())
    return T.ToPILImage()(res)