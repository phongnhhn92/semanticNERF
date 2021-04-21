import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image

def save_depth(v,cmap=cv2.COLORMAP_JET):
    depth = np.asarray(v, dtype=np.float32)
    depth = depth / depth.max()
    depth = np.asarray(depth * 255, np.uint8)
    depth = Image.fromarray(cv2.applyColorMap(depth, cmap))
    depth = T.ToTensor()(depth) # (3, H, W)
    return depth

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/max(ma-mi, 1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

