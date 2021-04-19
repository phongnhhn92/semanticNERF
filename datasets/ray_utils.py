import torch
from einops import repeat
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
        torch.stack([(i - W / 2) / focal, -(j - H / 2) / focal, -torch.ones_like(i)], -1)  # (H, W, 3)

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


def getRandomRays(hparams, data, semantics_nv, alpha_nv, style_code):
    all_rgb_gt = []
    all_rays = []
    all_semantics = []
    all_alphas = []
    all_styles = []
    SB, _, H, W = data['input_seg'].shape
    for target_rays, target_rays_gt, semantics_nv_b, alpha_nv_b, style_code_b \
            in zip(data['target_rays'], data['target_rgb_gt'], semantics_nv, alpha_nv, style_code):
        # Conver rgb values from 0 to 1
        target_rays_gt = target_rays_gt * 0.5 + 0.5

        rays_style_code = repeat(style_code_b.unsqueeze(0), '1 n1 -> r n1', r=hparams.num_rays)

        # Randomly sample a few rays in the target view.
        pix_inds = torch.randint(0, target_rays.shape[0], (hparams.num_rays,))
        rays = target_rays[pix_inds]
        rays_gt = target_rays_gt[pix_inds]
        rays_semantics = semantics_nv_b[pix_inds]
        rays_alphas = alpha_nv_b[pix_inds]

        all_rgb_gt.append(rays_gt)
        all_rays.append(rays)
        all_semantics.append(rays_semantics)
        all_alphas.append(rays_alphas)
        all_styles.append(rays_style_code)


    all_rgb_gt = torch.stack(all_rgb_gt).view(-1, 3)  # (SB * num_rays, 3)
    all_rays = torch.stack(all_rays).view(-1, 6)  # (SB * num_rays, 6)
    all_semantics = torch.stack(all_semantics).view(-1, _)
    all_alphas = torch.stack(all_alphas).view(-1, hparams.num_planes)
    all_styles = torch.stack(all_styles).view(-1, hparams.style_feat)

    return all_rgb_gt, all_rays, all_semantics, all_alphas, all_styles
