from .carla_utils.utils import *
from torchvision import transforms as T
from torch.utils.data import Dataset
from .ray_utils import *
import torch.nn.functional as F

def normalize(v):
    """Normalize a vector."""
    return v/np.linalg.norm(v)

def create_spiral_poses(radii, focus_depth, n_poses=120):
    """
    Computes poses that follow a spiral path for rendering purpose.
    See https://github.com/Fyusion/LLFF/issues/19
    In particular, the path looks like:
    https://tinyurl.com/ybgtfns3

    Inputs:
        radii: (3) radii of the spiral for each axis
        focus_depth: float, the depth that the spiral poses look at
        n_poses: int, number of poses to create along the path

    Outputs:
        poses_spiral: (n_poses, 3, 4) the poses in the spiral path
    """

    poses_spiral = []
    for t in np.linspace(0, 4 * np.pi, n_poses + 1)[:-1]:  # rotate 4pi (2 rounds)
        # the parametric function of the spiral (see the interactive web)
        center = np.array([np.cos(t), -np.sin(t), -np.sin(0.5 * t)]) * radii

        # the viewing z axis is the vector pointing from the @focus_depth plane
        # to @center
        z = normalize(center - np.array([0, 0, -focus_depth]))

        # compute other axes as in @average_poses
        y_ = np.array([0, 1, 0])  # (3)
        x = normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)

        poses_spiral += [np.stack([x, y, z, center], 1)]  # (3, 4)

    return np.stack(poses_spiral, 0)  # (n_poses, 3, 4)

def one_hot_encoding(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''
    one_hot = torch.FloatTensor(labels.size(0), C, labels.size(2), labels.size(3)).to(labels.device).zero_()
    target = one_hot.scatter_(1, labels.data, 1)

    return target

class CarlaDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 600), spheric_poses=False, val_num=1,classes = 13):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.spheric_poses = spheric_poses
        self.val_num = max(1, val_num)
        self.classes = classes

        # For rendering
        self.type = 'spiral'
        #self.type = 'horizontal'

        assert self.split != 'train' or self.split != 'val', 'Not implemented yet'
        self.define_transforms()
        self.read_meta()
        self.white_back = False

    def read_meta(self):
        n = 7  # 7x7 light field camera
        self.list_pose = []
        self.list_data = []
        self.all_rgbs = []
        self.all_segs = []

        for i in range(n):
            for j in range(n):
                data = read_cam(self.root_dir, i, j)
                self.list_data.append(data)
                # Get focal length, all images share same focal length
                if i == 0 and i == 0:
                    self.focal = np.asscalar(data['k'][0, 0].numpy())

                pose = data['pose']
                img = data['rgb'].view(3, -1).permute(1, 0)

                # Semantic value from 0-12 -> 13 classes
                semantic = data['sem'].unsqueeze(0)
                semantic = semantic.long()
                semantic = one_hot_encoding(semantic,C = self.classes).squeeze(0)
                semantic = semantic.view(self.classes,-1).permute(1,0)

                self.all_rgbs += [img]
                self.all_segs += [semantic]
                self.list_pose.append(pose)

        self.list_pose = torch.stack(self.list_pose)

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.img_wh[1], self.img_wh[0], self.focal)  # (H, W, 3)

        # correct poses
        # poses rotation should be "right up back"
        # See https://github.com/bmild/nerf/issues/34
        # This function might not be nesscesary, check later
        #self.poses, self.pose_avg = center_poses(self.list_pose)

        if self.split == 'train':
            self.all_rays = []
            for i, p in enumerate(self.list_pose):
                # Get rays
                rays_o, rays_d = get_rays(self.directions, torch.FloatTensor(p))

                if not self.spheric_poses:
                    near, far = 0, 1
                    rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                                  self.focal, 1.0, rays_o, rays_d)
                else:
                    print('No spheric_poses yet, only support forward-facing scene now. 360-scene later.')

                self.all_rays += [torch.cat([rays_o, rays_d,
                                             near * torch.ones_like(rays_o[:, :1]),
                                             far * torch.ones_like(rays_o[:, :1])],
                                            1)]  # (h*w, 8)

            assert len(self.all_rays) == len(self.all_rgbs) and len(self.all_rgbs) == len(self.all_segs), \
                'Mismatch number of rays and rgb'
            self.all_rays = torch.cat(self.all_rays, 0)  # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # ((N_images-1)*h*w, 3)
            self.all_segs = torch.cat(self.all_segs, 0)  # ((N_images-1)*h*w, 1)
        elif self.split == 'val':
            print('val image number is', n ** 2 // 2)
            self.val_idx = n ** 2 // 2
        else:
            if self.split.endswith('train'): # test on training set
                self.poses_test = self.list_pose
            elif not self.spheric_poses:
                if self.type == 'spiral':
                    focus_depth = 3.5 # hardcoded, this is numerically close to the formula
                                      # given in the original repo. Mathematically if near=1
                                      # and far=infinity, then this number will converge to 4
                    radii = np.percentile(np.abs(self.list_pose[..., 3]), 90, axis=0)
                    self.poses_test = create_spiral_poses(radii, focus_depth,n_poses=120)
                elif self.type == 'horizontal':
                    # Hard-coded since we know the x axis span between -0.3 and 0.3
                    arr = np.linspace(-1, 1, 50)
                    self.poses_test = []
                    for a in arr:
                        temp_p = self.list_pose[0].clone()
                        temp_p[0,3] = a
                        self.poses_test.append(temp_p)
                    self.poses_test = torch.stack(self.poses_test)


    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return self.val_num
        if self.split == 'test_train':
            return len(self.list_pose)
        return len(self.poses_test)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx],
                      'segs': self.all_segs[idx]}

            return sample
        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.list_pose[self.val_idx])
            elif self.split == 'test_train':
                c2w = torch.FloatTensor(self.list_pose[idx])
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])

            rays_o, rays_d = get_rays(self.directions, c2w)

            if not self.spheric_poses:
                near, far = 0, 1
                rays_o, rays_d = get_ndc_rays(self.img_wh[1], self.img_wh[0],
                                              self.focal, 1.0, rays_o, rays_d)
            else:
                print('No spheric_poses yet, only support forward-facing scene now. 360-scene later.')

            rays = torch.cat([rays_o, rays_d,
                              near * torch.ones_like(rays_o[:, :1]),
                              far * torch.ones_like(rays_o[:, :1])],
                             1)  # (h*w, 8)

            sample = {'rays': rays,
                      'c2w': c2w}

            if self.split in ['val', 'test_train']:
                if self.split == 'val':
                    idx = self.val_idx
                img = self.list_data[idx]['rgb'].view(3, -1).permute(1, 0)
                sample['rgbs'] = img

            return sample


    def define_transforms(self):
        self.transform = T.ToTensor()
