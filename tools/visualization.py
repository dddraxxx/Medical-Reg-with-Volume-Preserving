import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch

def identify_axes(ax_dict, fontsize=48):
    kw = dict(ha="center", va="center", fontsize=fontsize, color="darkgrey")
    for k, ax in ax_dict.items():
        ax.text(0.5, 0.5, k, transform=ax.transAxes, **kw)

def gca3D(**kwargs):
    fig = plt.figure(figsize=(4,4), **kwargs)
    return Axes3D(fig)

def plt_ffd(ffd, interval=1):
    fig = plt.figure(figsize=(4, 4))
    ax = Axes3D(fig)
    if isinstance(interval, int):
        sl = np.s_[::interval, ::interval, ::interval]
    else:
        sl = interval
    intrv = lambda x:x.reshape(*ffd.n_control_points, 3)[sl].reshape(-1,3)
    ax.scatter(*intrv(ffd.control_points(False)).T, s=10, c='blue')
    ax.scatter(*intrv(ffd.control_points()).T, s=5, c='red')
    # plt_grid3d(ffd.control_points(False), ax, colors='blue')
    plt_grid3d(intrv(ffd.control_points()), ax, colors='red', linewidths=0.5)
    plt.show()

def plt_grid3d(pts, ax, **kwargs):
    l = np.round(len(pts)**(1/3)).astype(int)
    x, y, z = pts.T.reshape(-1, l, l, l)
    grid1 = np.stack((x,y,z), axis=-1)
    ax.add_collection3d(Line3DCollection(grid1.reshape(-1, l, 3), **kwargs))
    ax.add_collection3d(Line3DCollection(grid1.transpose(0,2,1,3).reshape(-1, l, 3),  **kwargs))
    ax.add_collection3d(Line3DCollection(grid1.transpose(1,2,0,3).reshape(-1, l, 3),  **kwargs))

def plot_grid(ax, flow, factor=10):
    """
    Plot the grid generated by a flow on the given axis.

    Parameters:
    ax (matplotlib.axes.Axes): The axis to plot the grid on.
    flow (numpy.ndarray, H, W, C): A 2D array representing the flow.
    factor (int, optional): Scale factor to amplify the displacement of the flow. Default is 10.

    Returns:
    None
    """
    grid = factor * flow
    lin_range = np.arange(0, grid.shape[0], 1)
    x, y = np.meshgrid(lin_range, lin_range, indexing='ij')
    x = x + grid[..., 0]
    y = y + grid[..., 1]
    y = y

    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, color='black', linewidths=0.8))
    ax.add_collection(LineCollection(segs2, color='black', linewidths=0.8))
    ax.autoscale()

def plt_close():
    plt.close()

def plot_landmarks(img, landmarks, fig=None, ax=None, save_path=None, every_n = 3, color='red',size=10):
    """
    Plot landmarks on the input image.

    Parameters:
    - img (torch.tensor): Input image tensor.
    - landmarks (torch.tensor): Input landmarks tensor.
    - fig (matplotlib.figure.Figure, optional): Figure object to be used for plotting. Default is None.
    - ax (matplotlib.axes._subplots.AxesSubplot, optional): Axes object to be used for plotting. Default is None.
    - save_path (str, optional): Path to save the plot. Default is None.
    - every_n (int, optional): Plot every n-th slice of the input image. Default is 3.
    - color (str, optional): Color of the landmarks. Default is 'red'.
    - size (int, optional): Size of the landmarks in the plot. Default is 10.

    Returns:
    - fig (matplotlib.figure.Figure): Figure object used for plotting.
    - axes (matplotlib.axes._subplots.AxesSubplot): Axes object used for plotting.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
        landmarks = landmarks.cpu().numpy()
    if fig is None:
        fig = plt.figure(figsize=(10, 10))
    if ax is None:
        axes = fig.subplots(len(landmarks), 5)
    else: axes = ax
    # calculate idx to be visualized
    for i in range(landmarks.shape[0]):
        x0 = landmarks[i, 0]
        axes[i,2].scatter(landmarks[i, 2], landmarks[i, 1], s=size, c=color, marker='x')
        if ax is None:
            axes[i,0].imshow(img[int(x0)-2*every_n, ...], cmap='gray')
            # caption
            axes[i,0].set_title(f'{int(x0)-2*every_n}')
            axes[i,1].imshow(img[int(x0)-every_n, ...], cmap='gray')
            axes[i,1].set_title(f'{int(x0)-every_n}')
            axes[i,2].imshow(img[int(x0), ...], cmap='gray')
            axes[i,2].set_title(f'{int(x0)}, ({landmarks[i, 0]}, {landmarks[i, 1]}, {landmarks[i, 2]})')
            axes[i,3].imshow(img[int(x0)+every_n, ...], cmap='gray')
            axes[i,3].set_title(f'{int(x0)+every_n}')
            axes[i,4].imshow(img[int(x0)+2*every_n, ...], cmap='gray')
            axes[i,4].set_title(f'{int(x0)+2*every_n}')
    # tight layout and no axis
    fig.tight_layout()
    for ax in fig.get_axes():
        ax.axis('off')
    if save_path is not None:
        plt.savefig(save_path)
    return fig, axes

def plot_to_PIL():
    """
    Convert a matplotlib plot into a PIL image."""
    from io import BytesIO
    from PIL import Image
    # convert the plot into a bytes-like object
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    # open the image using PIL
    im = Image.open(buf)
    return im


if __name__=='__main__':
    #%%
    import monai
    file = '/mnt/sdc/lits/train/volume-102.nii'
    fdir = '/mnt/sdc/lits/train'
    # glob nii
    import glob
    nii_files = sorted(glob.glob(f'{fdir}/volum*.nii'), key=lambda x: int(x.split('-')[-1].split('.')[0]))
    print(nii_files)
    #%%
    from monai.transforms import (
        Compose, LoadImage
    )
    print(nii_files[128])
    LoadImage()(nii_files[128])
    #%%
    img, meta = LoadImage()(nii_files[75])
    meta['space']