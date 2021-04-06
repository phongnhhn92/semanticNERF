import torch
from einops import rearrange, reduce, repeat

__all__ = ['render_rays']


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0,
                         # in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(model,
                embeddings,
                rays,
                radius,
                N_samples=64,
                use_disp=False,
                perturb=0,
                noise_std=1,
                N_importance=0,
                chunk=1024*32,
                white_back=False,
                test_time=False,
                **kwargs
                ):
    """
    Render rays by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3+2), ray origins and directions, near and far depths
        N_samples: number of coarse samples per ray
        use_disp: whether to sample in disparity space (inverse depth)
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        white_back: whether the background is white (dataset dependent)
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, typ, xyz, z_vals, test_time=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            typ: 'coarse' or 'fine'
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
            z_vals: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        Outputs:
            if weights_only:
                weights: (N_rays, N_samples_): weights of each sample
            else:
                rgb_final: (N_rays, 3) the final rgb image
                depth_final: (N_rays) depth map
                weights: (N_rays, N_samples_): weights of each sample
        """
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c') # (N_rays*N_samples_, 3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        if typ=='coarse' and test_time:
            for i in range(0, B, chunk):
                xyzd_IPE_chunk = xyz_[i:i+chunk]
                out_chunks += [model(xyzd_IPE_chunk, sigma_only=True)]

            out = torch.cat(out_chunks, 0)
            sigmas = rearrange(out, '(n1 n2) 1 -> n1 n2', n1=N_rays, n2=N_samples_)
        else: # infer rgb and sigma and others
            for i in range(0, B, chunk):
                xyzd_IPE_chunk = xyz_[i:i+chunk]
                out_chunks += [model(xyzd_IPE_chunk, sigma_only=False)]

            out = torch.cat(out_chunks, 0)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_, c=4)
            rgbs = out[..., :3] # (N_rays, N_samples_, 3)
            sigmas = out[..., 3] # (N_rays, N_samples_)
            
        # Convert these values using volume rendering (Section 4)
        cone_midpoints = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
        #deltas = cone_midpoints[:, 1:] - cone_midpoints[:, :-1]  # (N_rays, N_samples_-1)
        deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
        #delta_inf = 1e10 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        #deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        # compute alpha by the formula (3)
        noise = torch.randn_like(sigmas) * noise_std
        #alphas = 1-torch.exp(-deltas*torch.relu(sigmas+noise)) # (N_rays, N_samples_)

        # softplus: f(x) = ln(1+e^x)
        alphas = 1 - torch.exp(-deltas * torch.nn.Softplus()(sigmas + noise))  # (N_rays, N_samples_)

        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas+1e-10], -1) # [1, 1-a1, 1-a2, ...]
        weights = \
            alphas * torch.cumprod(alphas_shifted[:, :-1], -1) # (N_rays, N_samples_)
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum') # (N_rays), the accumulated opacity along the rays
                                                            # equals "1 - (1-a1)(1-a2)...(1-an)" mathematically

        results[f'weights_{typ}'] = weights
        results[f'opacity_{typ}'] = weights_sum
        results[f'z_vals_{typ}'] = z_vals
        if test_time and typ == 'coarse':
            return

        rgb_map = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*rgbs, 'n1 n2 c -> n1 c', 'sum')
        depth_map = reduce(weights*cone_midpoints, 'n1 n2 -> n1', 'sum')

        if white_back:
            rgb_map += 1-weights_sum.unsqueeze(1)

        results[f'rgb_{typ}'] = rgb_map
        results[f'depth_{typ}'] = depth_map

        return

    def cast_cone(origin, direction, t):
        """
        https://arxiv.org/abs/2103.13415
        Cone casting
        Approximate the conial frustum with a multivariate Gaussian which is charaterized by :
        1) mean distance along the ray t_mean
        2) variance along the ray t_var
        3) variance perpendicular to the ray r_var
        """
        t0, t1 = t[:, :-1], t[:, 1:]
        c, d = (t0 + t1) / 2, (t1 - t0) / 2
        t_mean = c + (2 * c * d ** 2) / (3 * c ** 2 + d ** 2)
        t_var = (d ** 2) / 3 - (4 / 15) * ((d ** 4 * (12 * c ** 2 - d ** 2)) / (3 * c ** 2 + d ** 2) ** 2)
        rad = repeat(radius, 'b -> b c', c=c.shape[-1]).type_as(c)
        r_var = rad ** 2 * ((c ** 2) / 4 + (5 / 12) * d ** 2 - (4 / 15) * (d ** 4) / (3 * c ** 2 + d ** 2))
        mean = direction[..., None, :] * t_mean[..., None]
        null_outer_diag = 1 - (rays_d ** 2) / torch.sum(direction ** 2, -1, keepdim=True)
        cov_diag = (t_var[..., None] * (rays_d ** 2)[..., None, :]
                    + r_var[..., None] * null_outer_diag[..., None, :])

        return mean + origin[..., None, :],cov_diag

    # Decompose the inputs
    N_rays = rays.shape[0]
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8] # both (N_rays, 1)

    # Sample depth points
    z_steps = torch.linspace(0, 1, N_samples+1, device=rays.device) # (N_samples)
    if not use_disp: # use linear sampling in depth space
        z_vals = near * (1-z_steps) + far * z_steps
    else: # use linear sampling in disparity space
        z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)

    if perturb > 0: # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[: ,-1:]], -1)
        lower = torch.cat([z_vals[: ,:1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    mean, cov_diag = cast_cone(rays_o,rays_d,z_vals)

    # IPE input to MLP
    IPEmbeded_coarse = embeddings(mean,cov_diag)
    results = {}
    inference(results, model, 'coarse', IPEmbeded_coarse, z_vals, test_time, **kwargs)

    if N_importance > 0: # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:]) # (N_rays, N_samples-1) interval mid points

        # 2-tap max filter
        #w_k,w_k_prev,w_k_next = results['weights_coarse'][:,1:-1], results['weights_coarse'][:,:-2], \
        #                            results['weights_coarse'][:,2:]
        # alpha  = 0.01
        #weight_c = 0.5 * (torch.maximum(w_k_prev,w_k)+ torch.maximum(w_k,w_k_next)) + 0.01

        z_vals_ = sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1],
                             N_importance, det=(perturb==0))
                  # detach so that grad doesn't propogate to weights_coarse from here

        z_vals = torch.sort(z_vals_, -1)[0]
        mean, cov_diag = cast_cone(rays_o, rays_d, z_vals)
        # IPE input to MLP
        IPEmbeded_fine = embeddings(mean, cov_diag)
        inference(results, model, 'fine', IPEmbeded_fine, z_vals, test_time, **kwargs)

    return results
