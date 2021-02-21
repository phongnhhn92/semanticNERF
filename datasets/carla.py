from .carla_utils.utils import *
from torchvision import transforms as T
from torch.utils.data import Dataset
from .carla_utils.utils import *


class CarlaDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(800, 600), spheric_poses=False, val_num=1):
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

        assert self.split != 'train' or self.split != 'val', 'Not implemented yet'
        self.define_transforms()
        self.read_meta()
        self.white_back = False

    def read_meta(self):
        n = 7  # 7x7 light field camera
        data_path = self.root_dir
        self.list_pose = []
        self.all_rgbs = []
        self.list_data = []
        for i in range(n):
            for j in range(n):
                data = read_cam(data_path, i, j)
                self.list_data.append(data)
                # Get focal length, all images share same focal length
                if i == 0 and i == 0:
                    self.focal = np.asscalar(data['k'][0, 0].numpy())

                pose = data['pose']
                img = data['rgb'].view(3, -1).permute(1, 0)
                self.all_rgbs += [img]
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

            assert len(self.all_rays) != self.all_rgbs, 'Mismatch number of rays and rgb'
            self.all_rays = torch.cat(self.all_rays, 0)  # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # ((N_images-1)*h*w, 3)
        else:
            print('val image number is', n ** 2 // 2)
            self.val_idx = n ** 2 // 2

    def __len__(self):
        if self.split == 'train':
            return len(self.all_rays)
        if self.split == 'val':
            return self.val_num
        # if self.split == 'test_train':
        #     return len(self.poses)
        return len(self.all_rays)

    def __getitem__(self, idx):

        if self.split == 'train':  # use data in the buffers
            sample = {'rays': self.all_rays[idx],
                      'rgbs': self.all_rgbs[idx]}

            return sample
        elif self.split == 'val':
            c2w = torch.FloatTensor(self.list_pose[self.val_idx])
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

            img = self.list_data[self.val_idx]['rgb'].view(3, -1).permute(1, 0)
            sample['rgbs'] = img

            return sample


    def define_transforms(self):
        self.transform = T.ToTensor()
