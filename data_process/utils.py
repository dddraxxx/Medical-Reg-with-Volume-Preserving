import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label

def dict2np(d):
    """ convert dict to numpy array """
    return np.array([d[k] for k in sorted(d.keys(), key=lambda x: int(x))])

def getLargestCC(segmentation):
    """ get the largest connected component """
    labels = label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

# crop seg by the bounding box where value is larger than 0
def bbox_seg(seg):
    """ crop the seg to the bbox

    Returns:
        bbox: [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    bbox = seg.nonzero()
    bbox = np.array([np.min(bbox[0]), np.min(bbox[1]), np.min(
        bbox[2]), np.max(bbox[0]), np.max(bbox[1]), np.max(bbox[2])])
    return bbox

def draw_img_point(img, pt):
    fig = plt.figure()
    ax = fig.gca()

    # draw the point
    pt = np.round(pt).astype(int)
    ax.scatter(pt[2], pt[1], c='r', s=10)

    # draw the image
    img = img[pt[0], :, :]
    ax.imshow(img, cmap='gray')
    return ax
