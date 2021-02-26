import torch,os,math
import numpy as np
import cv2 as cv
from PIL import Image
import torch.nn.functional as F

carla_pallete = {
    0: [0, 0, 0],  # None
    1: [70, 70, 70],  # Buildings
    2: [190, 153, 153],  # Fences
    3: [72, 0, 90],  # Other
    4: [220, 20, 60],  # Pedestrians
    5: [153, 153, 153],  # Poles
    6: [157, 234, 50],  # RoadLines
    7: [128, 64, 128],  # Roads
    8: [244, 35, 232],  # Sidewalks
    9: [107, 142, 35],  # Vegetation
    10: [0, 0, 255],  # Vehicles
    11: [102, 102, 156],  # Walls
    12: [220, 220, 0],  # TrafficSigns
    13: [150, 33, 88],  # TrafficSigns
    14: [111, 74,  0],
    15: [81, 0, 81],
    16: [250, 170, 160],
    17: [230, 150, 140],
    18: [180, 165, 180],
    19: [150, 100, 100],
    20: [150, 120, 90],
    21: [250, 170, 30],
    22: [220, 220,  0],
    23: [152, 251, 152],
    24: [70, 130, 180],
    25: [255, 0, 0],
    26: [0, 0, 142],
    27: [0, 0, 70],
    28: [0, 60, 100],
    29: [0, 0, 110],
    20: [0, 80, 100],
    31: [0, 0, 230],
    32: [119, 11, 32],
}

def get_palette(dataset_name):
    pallet_map = {}
    pallet_map['carla'] = carla_pallete
    assert dataset_name in pallet_map.keys(
    ), f'Unknown dataset {dataset_name}: not in {pallet_map.keys()} '
    return pallet_map[dataset_name]

def get_num_classes(dataset_name):
    num_classes_map = {}
    num_classes_map['carla'] = 13
    # num_classes_map['scan_net'] = 13
    # num_classes_map['scenenet_rgbd'] = 14
    num_classes_map['vkitti'] = 16
    num_classes_map['cityscapes'] = 20
    assert dataset_name in num_classes_map.keys(
    ), f'Unknown dataset {dataset_name}: not in {num_classes_map.keys()} '
    return num_classes_map[dataset_name]

class SaveSemantics:
    '''
    Currently supports the following datasets
    ['carla', 'vkitti', 'cityscapes']
    '''

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.num_classes = get_num_classes(dataset_name)
        self.pallete = get_palette(dataset_name)

    def __call__(self, input_seg, file_name):
        input_seg = input_seg.squeeze()
        assert input_seg.ndimension(
        ) == 2, 'input segmentation should be either [H, W] or [1, H, W]'
        self.save_lable(input_seg, file_name)

    def to_color(self, input_seg):
        assert self.num_classes > input_seg.max(), 'Segmentaion mask > num_classes'
        input_seg = input_seg.int().squeeze().numpy()
        seg_mask = np.asarray(input_seg, dtype=np.uint8)
        pil_im = Image.fromarray(seg_mask, mode="P")
        pallette_ = []
        for v in self.pallete.values():
            pallette_.extend(v)
        for _i in range(len(self.pallete.keys()), 256):
            pallette_.extend([0, 0, 0])
        pil_im.putpalette(pallette_)
        pil_np = np.asarray(pil_im.convert('RGB'), dtype=np.uint8)
        return pil_np

    def save_lable(self, input_seg, file_name):
        col_img = self.to_color(input_seg)
        pil_im = Image.fromarray(col_img)
        pil_im.save(file_name)

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
    pose_file = os.path.join(data_path, 'cam_2_world.txt')
    # As the world is arbitrary  we need to set the center camera as the reference
    pose_data = np.loadtxt(pose_file)
    pose_data = pose_data.reshape([lf_size**2, 3, 4])
    ref_pose = pose_data[(lf_size*lf_size)//2]
    r_ref, t_ref = ref_pose[:3, :3], ref_pose[:3, 3]
    # Change from carla cooridnates to ours
    # carla x-forward, y-right, z-up
    # cours x-right, y-down, z-forward
    #r_carla_2_ours = np.array([[0, 1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)

    # Teddy check
    # Since I change this, the depth-based warping function in reader.py might not work anymore
    # NERF  x-right, y-up,   z-back
    r_carla_2_ours = np.array([[0, 1, 0], [0, 0, 1], [-1, 0, 0]], dtype=np.float32)
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
    col_file = os.path.join(data_path, 'RGB_{:0>2d}_{:0>2d}.png'.format(row, col))
    dep_file = os.path.join(data_path, 'Depth_{:0>2d}_{:0>2d}.png'.format(row, col))
    sem_file = os.path.join(data_path, 'SEM_{:0>2d}_{:0>2d}.png'.format(row, col))
    pose_data = read_pose(data_path)
    pose = pose_data[int(row*lf_size + col)]
    col_img = torch.from_numpy(np.asarray(Image.open(col_file))).float().div(255.0)
    col_img = col_img.permute(2, 0, 1)
    sem_img = read_sem(sem_file)
    dep_img = read_dep(dep_file)
    k_matrix = get_k()
    r_mat = pose[:3, :3]
    t_vec = (pose[:3, 3]).view(3,1)
    data = {'rgb':col_img, 'sem':sem_img, 'dep':dep_img, 'k':k_matrix, 'r': r_mat, 't':t_vec,'pose':pose}
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