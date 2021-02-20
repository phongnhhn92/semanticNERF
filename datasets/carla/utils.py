import torch,os,math
import numpy as np
import cv2 as cv
from PIL import Image
import torch.nn.functional as F

def get_k(fov=90.0, height=600, width=800):
    k = np.identity(3)
    k[0, 2] = width / 2.0
    k[1, 2] = height / 2.0
    k[0, 0] = k[1, 1] = width / \
        (2.0 * math.tan(fov * math.pi / 360.0))
    return torch.from_numpy(k)

def read_dep(depth_path):
    img = np.asarray(Image.open(depth_path), dtype=np.uint8)
    img = img.astype(np.float64)[:,:,:3]
    normalized_depth = np.dot(img, [1.0, 256.0, 65536.0])
    normalized_depth /= 16777215.0
    normalized_depth = torch.from_numpy(normalized_depth * 1000.0)
    return normalized_depth.unsqueeze(0)

def read_sem(semantics_path, height=None, width=None):
    seg = cv.imread(semantics_path, cv.IMREAD_ANYCOLOR |
                    cv.IMREAD_ANYDEPTH)
    seg = np.asarray(seg, dtype=np.uint8)
    seg = torch.from_numpy(seg[..., 2]).float().squeeze()
    h, w = seg.shape
    if (not height is None) and (not width is None):
        seg = F.interpolate(seg.view(1, 1, h, w), size=(height, width),
                            mode='nearest').squeeze()
    return seg.unsqueeze(0)

def read_pose(data_path, lf_size=7):
    pose_file = os.path.join(data_path, '000000', 'cam_2_world.txt')
    # As the world is arbitrary  we need to set the center camera as the reference
    pose_data = np.loadtxt(pose_file)
    pose_data = pose_data.reshape([lf_size**2, 3, 4])
    ref_pose = pose_data[(lf_size*lf_size)//2]
    r_ref, t_ref = ref_pose[:3, :3], ref_pose[:3, 3]
    # Change from carla cooridnates to ours
    # carla x-forward, y-right, z-up
    # cours x-right, y-down, z-forward
    r_carla_2_ours = np.array([[0, 1, 0], [0, 0, -1], [1,  0, 0]], dtype=np.float32)
    # r_carla_2_ours = np.eye(3)
    new_pose = []
    for pose in pose_data:
        # pose-> cam2world
        # return cam2ref
        r_cam, t_cam = pose[:3, :3], pose[:3, 3]
        # Rotation: from cam to reference: world2ref @ cam2world
        r_cam_2_r = r_ref.T @ r_cam
        # Translation: cam2 as seen from reference
        t_vec = r_ref.T @ (t_cam - t_ref)
        # convert rotation matrix to our coordinate system convention
        r_cam_2_r = r_carla_2_ours @ r_cam_2_r @ r_carla_2_ours.T
        #
        # convert translation vec to our coordinate system convention
        t_vec = r_carla_2_ours @ t_vec
        pose_cam_2_ref = np.hstack([r_cam_2_r, t_vec.reshape(3,1)])
        new_pose.append(torch.from_numpy(pose_cam_2_ref))
    return new_pose

def read_cam(data_path, row, col, lf_size=7):
    ''' give path to an episode and camera row and col this function returns
        RGB, SEM, DEPTH images, K matrix, extrinsic camera pose
    '''
    col_file = os.path.join(data_path, '000000', 'RGB_{:0>2d}_{:0>2d}.png'.format(row, col))
    dep_file = os.path.join(data_path, '000000', 'Depth_{:0>2d}_{:0>2d}.png'.format(row, col))
    sem_file = os.path.join(data_path, '000000', 'SEM_{:0>2d}_{:0>2d}.png'.format(row, col))
    pose_data = read_pose(data_path)
    pose = pose_data[int(row*lf_size + col)]
    col_img = torch.from_numpy(np.asarray(Image.open(col_file))).float().div(255.0)
    col_img = col_img.permute(2, 0, 1)
    sem_img = read_sem(sem_file)
    dep_img = read_dep(dep_file)
    k_matrix = get_k()
    r_mat = pose[:3, :3]
    t_vec = (pose[:3, 3]).view(3,1)
    data = {'rgb':col_img, 'sem':sem_img, 'dep':dep_img, 'k':k_matrix, 'r': r_mat, 't':t_vec}
    data = {k:v.float() for k,v in data.items()}
    return data

# functions for warping with depth
def get_grid(b,h,w, device='cpu'):
    x_locs, y_locs = torch.linspace(0, w-1, w).to(device), torch.linspace(0,h-1, h).to(device)
    x_locs, y_locs = x_locs.view(1, 1, w, 1), y_locs.view(1, h, 1, 1)
    x_locs, y_locs = x_locs.expand(b,h,w,1), y_locs.expand(b,h,w,1)
    xy_grid = torch.cat([x_locs, y_locs], 3)
    return xy_grid

def make_homogeneous(xy_grid):
    b,h,w,_ = xy_grid.shape
    device = xy_grid.device
    ones = torch.ones(b,h,w,1).to(device)
    return torch.cat([xy_grid, ones], 3)

def warp(sampling_grid, input_img):
    b, h, w, _ = sampling_grid.shape
    x_locs, y_locs = torch.split(sampling_grid, dim=3, split_size_or_sections=1)
    x_locs = (x_locs - ((w -1)/2))/((w -1)/2)
    y_locs = (y_locs - ((h -1)/2))/((h -1)/2)
    grid  = torch.cat([x_locs, y_locs], dim=3)
    return F.grid_sample(input=input_img, grid=grid, mode='bilinear', align_corners=True)