import torch,os,math
import numpy as np
import cv2 as cv
from PIL import Image
import torch.nn.functional as F
from kornia import create_meshgrid

def get_ray_directions(H, W, focal):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-W/2)/focal, -(j-H/2)/focal, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions


def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T  # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:, 3].expand(rays_d.shape)  # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def get_ndc_rays(H, W, focal, near, rays_o, rays_d):
    """
    Transform rays from world coordinate to NDC.
    NDC: Space such that the canvas is a cube with sides [-1, 1] in each axis.
    For detailed derivation, please see:
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf

    In practice, use NDC "if and only if" the scene is unbounded (has a large depth).
    See https://github.com/bmild/nerf/issues/18

    Inputs:
        H, W, focal: image height, width and focal length
        near: (N_rays) or float, the depths of the near plane
        rays_o: (N_rays, 3), the origin of the rays in world coordinate
        rays_d: (N_rays, 3), the direction of the rays in world coordinate

    Outputs:
        rays_o: (N_rays, 3), the origin of the rays in NDC
        rays_d: (N_rays, 3), the direction of the rays in NDC
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Store some intermediate homogeneous results
    ox_oz = rays_o[..., 0] / rays_o[..., 2]
    oy_oz = rays_o[..., 1] / rays_o[..., 2]

    # Projection
    o0 = -1. / (W / (2. * focal)) * ox_oz
    o1 = -1. / (H / (2. * focal)) * oy_oz
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - ox_oz)
    d1 = -1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - oy_oz)
    d2 = 1 - o2

    rays_o = torch.stack([o0, o1, o2], -1)  # (B, 3)
    rays_d = torch.stack([d0, d1, d2], -1)  # (B, 3)

    return rays_o, rays_d


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34

    Inputs:
        poses: (N_images, 3, 4)

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """

    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg

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