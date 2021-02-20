import torch,argparse
import torchvision
from utils import *

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='/media/phong/Data2TB/dataset/carla/carla/carla_phong_2/Town01/episode_00001/',
                        help='root directory of carla')

    return parser.parse_args()


if __name__ == '__main__':
    hparams = get_opts()
    # Depth based warping from camera b to to camera a
    # lets add a dummy batch dimension
    device = 'cuda:0'
    data_path = hparams.root_dir
    cam_a = read_cam(data_path, 3, 3)
    cam_b = read_cam(data_path, 5, 5)

    cam_a = {k:v.unsqueeze(0).to(device) for k,v in cam_a.items()}
    cam_b = {k:v.unsqueeze(0).to(device) for k,v in cam_b.items()}

    b,c,h,w = cam_a['rgb'].shape
    xy_locs = get_grid(b,h,w, device)
    xy_locs_h = make_homogeneous(xy_locs)
    k_inverse = torch.inverse(cam_a['k']).view(1, 1, 1, 3, 3)
    rays_cam_a = torch.matmul(k_inverse, xy_locs_h.unsqueeze(-1))
    depth = cam_a['dep'].view(b, h, w, 1, 1)
    pts_3d_cam_a = depth*rays_cam_a
    # Apply Extrinsics from camera A to camera B
    r_rel = torch.bmm(cam_b['r'].permute(0,2,1), cam_a['r'])
    t_rel = torch.bmm(cam_b['r'].permute(0,2,1), (cam_a['t'] - cam_b['t']))
    pts_3d_cam_b = torch.matmul(r_rel.view(1, 1, 1, 3, 3), pts_3d_cam_a) + t_rel.view(1, 1, 1, 3, 1)
    # Project camera p
    pts_2d_cam_b = torch.matmul(cam_b['k'].view(1, 1, 1, 3, 3), pts_3d_cam_b)
    pts_2d_cam_b[..., 0, 0] = pts_2d_cam_b[..., 0, 0] / pts_2d_cam_b[..., 2, 0]
    pts_2d_cam_b[..., 1, 0] = pts_2d_cam_b[..., 1, 0] / pts_2d_cam_b[..., 2, 0]
    # cample cam b color image using
    sampling_grid = (pts_2d_cam_b[:, :, :, :2, 0]).view(b, h, w, 2)
    cam_a['rgb_est'] = warp(sampling_grid, cam_b['rgb'])

    # print(rays_cam_a.shape)
    # print(cam_a['dep'].shape)

    torchvision.utils.save_image(cam_a['rgb_est'].cpu(), '0_warped_cam_a.png')
    torchvision.utils.save_image(cam_a['rgb'].cpu(), '1_real_cam_a.png')
    torchvision.utils.save_image(cam_b['rgb'].cpu(), '2_real_cam_b.png')