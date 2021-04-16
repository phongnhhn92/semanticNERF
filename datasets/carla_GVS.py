import os,csv,random,torch,math
from pathlib import Path
from PIL import Image
import cv2 as cv
import torch.nn.functional as F
import numpy as np
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms
from datasets.ray_utils import *

class CarlaGVSDataset(Dataset):
    def __init__(self, opts,split,numberOfRays = 128):
        super(CarlaGVSDataset,self).__init__()

        self.opts = opts
        self.split = split
        self.numberOfRays = numberOfRays
        self.camera_groups = ['ForwardCameras', 'SideCameras', 'HorizontalCameras']
        # Transformations
        self.to_tensor = transforms.Compose([transforms.ToTensor()])
        self.ToPIL = transforms.Compose([transforms.ToPILImage()])

        self.height, self.width = self.opts.img_wh[0], self.opts.img_wh[1]
        self.stereo_baseline = 0.54
        if self.split == 'train':
            self.base_path = f'{opts.root_dir}/{self.split}'
        else:
            self.base_path = f'{opts.root_dir}/test'
        assert os.path.exists(self.base_path), f'{self.base_path} folder does not exist'

        self.file_list = self.get_file_list(self.split)
        self.train_camera_suffix = [f'_{str(x).zfill(2)}' for x in range(5)]



    def get_file_list(self,split):
        if split == 'train':
            ''' For training we load a single view from a random camera group, town, weather condition and time step.
            A sample file in the returned list would look like
                 .../Town04/weather_03/HorizontalCameras/rgb/009990.png
            The source and target cameras will be decided inside the __getitem__()
            Example source and target views could be:
                .../Town00/weather_03/HorizontalCameras_03/rgb/009990.png
                .../Town00/weather_03/HorizontalCameras_00/rgb/009990.png
            '''
            episode_folders = [
                f'Town0{x}/weather_0{y}' for x in range(1, 6) for y in range(4)]

            join = lambda x:os.path.join(*x)
            file_list = [join([self.base_path, epi, cam, f'rgb/{str(x).zfill(6)}.png']) for epi in episode_folders
                            for cam in self.camera_groups \
                                for x in range(0, 10000, 10)]
            return file_list
        elif split == 'val':
            val_frames = []
            with open(os.path.join(self.opts.root_dir, 'test_samples.txt'), 'r') as fid:
                reader = list(csv.reader(fid))
                rnd_ind = random.randint(0,len(reader)-1)
                src = os.path.join(self.base_path, reader[rnd_ind][0])
                trg = os.path.join(self.base_path, reader[rnd_ind][1])
                val_frames.append([src, trg])
            return val_frames
        else:
            ''' For test we load list of source and target views
            In test phase the reference camera is at 00 and target it at 01 
            Example source and target views could be:
                .../Town04/weather_03/HorizontalCameras_00/rgb/000000.png
                .../Town04/weather_03/HorizontalCameras_01/rgb/000000.png
            '''
            test_frames = []
            with open(os.path.join(self.opts.data_path, 'carla_test_frames.txt'), 'r') as fid:
                reader = csv.reader(fid)
                for line in reader:
                    src = os.path.join(self.base_path, line[0])
                    trg = os.path.join(self.base_path, line[1])
                    test_frames.append([src, trg])
            return test_frames

    def __getitem__(self,index):
        if self.split=='train':
            sample = self.file_list[index]
            trg_cam, src_cam = random.sample(self.train_camera_suffix, 2)
            cam_group = Path(sample).parent.parent.stem
            # For more randomness
            # ind = random.randint(0,len(self.camera_groups)-1)
            # cam_group = self.camera_groups[ind]
            src_file = sample.replace(cam_group, cam_group+src_cam)
            trg_file = sample.replace(cam_group, cam_group+trg_cam)
        else:
            src_file, trg_file = self.file_list[index][0], self.file_list[index][1]
        input_img = self._read_rgb(src_file)
        target_img = self._read_rgb(trg_file)
        k_matrix = self._carla_k_matrix(height=self.height, width=self.width, fov=90)
        input_disp = self._read_disp(src_file.replace('rgb', 'depth'), k_matrix)
        target_disp = self._read_disp(trg_file.replace('rgb', 'depth'), k_matrix)
        input_seg = self._read_seg(src_file.replace('rgb', 'semantic_segmentation'))
        target_seg = self._read_seg(trg_file.replace('rgb', 'semantic_segmentation'))
        r_mat, t_vec = self._get_rel_pose(src_file, trg_file)
        input_pose = self._get_abs_pose(src_file)
        target_pose = self._get_abs_pose(trg_file)
        data_dict = {}
        data_dict['input_img'] = input_img
        data_dict['input_seg'] = input_seg
        data_dict['input_pose'] = input_pose
        data_dict['input_disp'] = input_disp
        data_dict['target_img'] = target_img
        data_dict['target_seg'] = target_seg
        data_dict['target_pose'] = target_pose
        data_dict['target_disp'] = target_disp
        data_dict['k_matrix'] = k_matrix
        data_dict['t_vec'] = t_vec
        data_dict['r_mat'] = r_mat
        data_dict['stereo_baseline'] = torch.Tensor([self.stereo_baseline])
        # Load style image, if passed, else the input will serve as style
        data_dict['style_img'] = input_img.clone()

        # Sample training rays of the target pose
        focal = k_matrix[0, 0]
        directions = get_ray_directions(self.height, self.width, focal)
        rays_o, rays_d = get_rays(directions, target_pose)
        # Use NDC for now
        # near, far = 0, 1
        # rays_o, rays_d = get_ndc_rays(self.height, self.width,
        #                               focal, 1.0, rays_o, rays_d)
        # Teddy: If we dont use NDC then I need to know the near and far depth plane.
        # Can we find it based on the GT target_disp ?
        cam_rays = torch.cat([rays_o, rays_d],1)
        rays_rgb = input_img.view(3,-1).permute(1,0)

        data_dict['target_rays'] = cam_rays
        data_dict['target_rgb_gt'] = rays_rgb

        data_dict = {k: v.float()
                     for k, v in data_dict.items() if not (k is None)}
        return data_dict

    def _get_t_vec(self,cam,id):
        if cam.startswith('ForwardCameras'):
            t_vec = [ 0, 0, (id - 2)*self.stereo_baseline]
        elif cam.startswith('HorizontalCameras'):
            t_vec = [ (id - 2)*self.stereo_baseline, 0 ,0]
        elif cam.startswith('SideCameras'):
            t_vec = [ (id - 2)*self.stereo_baseline, 0 ,0]
        return t_vec

    def _get_abs_pose(self,file):
        cam = Path(file).parent.parent.stem
        idx = int(cam[-2:])

        r_mat = torch.eye(3).float()
        t_vec = self._get_t_vec(cam, idx)
        t_vec = torch.FloatTensor(t_vec).view(3, 1)
        return torch.cat([r_mat,t_vec],dim=1)

    def _get_rel_pose(self, src_file, trg_file):
        cam_src = Path(src_file).parent.parent.stem
        cam_trg = Path(trg_file).parent.parent.stem
        src_idx, trg_idx = int(cam_src[-2:]), int(cam_trg[-2:])

        if cam_src.startswith('ForwardCameras'):
            x, y = 0, 0
            z = (src_idx - trg_idx)*self.stereo_baseline
        elif cam_src.startswith('HorizontalCameras'):
            y, z = 0, 0
            x = (src_idx - trg_idx)*self.stereo_baseline
        elif cam_src.startswith('SideCameras'):
            y, z = 0, 0
            x = (trg_idx - src_idx)*self.stereo_baseline
        else:
            assert False, f'unknown camera identifier {cam_src}'

        t_vec = torch.FloatTensor([x, y, z]).view(3, 1)
        r_mat = torch.eye(3).float()
        return r_mat, t_vec

    def _read_depth(self, depth_path):
        img = np.asarray(Image.open(depth_path), dtype=np.uint8)
        img = img.astype(np.float64)[:,:,:3]
        normalized_depth = np.dot(img, [1.0, 256.0, 65536.0])
        normalized_depth /= 16777215.0
        normalized_depth = torch.from_numpy(normalized_depth * 1000.0)
        return normalized_depth

    def _read_disp(self, depth_path, k_matrix):
        depth_img = self._read_depth(depth_path).squeeze()
        disp_img = self.stereo_baseline * (k_matrix[0, 0]).view(1, 1) / (depth_img.clamp(min=1e-06)).squeeze()
        h, w = disp_img.shape[:2]
        disp_img = disp_img.view(1, 1, h, w)
        disp_img = F.interpolate(disp_img, size=(self.height, self.width),
                                mode='bilinear', align_corners=False)
        disp_img = disp_img.view(1, self.height, self.width)
        return disp_img

    def __len__(self):
        return len(self.file_list)

    def label_to_one_hot(self, input_seg, num_classes=13):
        assert input_seg.max() < num_classes, f'Num classes == {input_seg.max()} exceeds {num_classes}'
        b, _, h, w = input_seg.shape
        lables = torch.zeros(b, num_classes, h, w).float()
        labels = lables.scatter_(dim=1, index=input_seg.long(), value=1.0)
        labels = labels.to(input_seg.device)
        return labels

    def _read_seg(self, semantics_path):
        seg = cv.imread(semantics_path, cv.IMREAD_ANYCOLOR |
                        cv.IMREAD_ANYDEPTH)
        seg = np.asarray(seg, dtype=np.uint8)
        seg = torch.from_numpy(seg[..., 2]).float().squeeze()
        h, w = seg.shape
        seg = F.interpolate(seg.view(1, 1, h, w), size=(self.height, self.width),
                            mode='nearest')
        # Change semantic labels to one-hot vectors
        seg = self.label_to_one_hot(seg, self.opts.num_classes).squeeze(0)
        return seg

    def _carla_k_matrix(self, fov=90.0, height=256, width=256):
        k = np.identity(3)
        k[0, 2] = width / 2.0
        k[1, 2] = height / 2.0
        k[0, 0] = k[1, 1] = width / \
            (2.0 * math.tan(fov * math.pi / 360.0))
        return torch.from_numpy(k)

    def _read_rgb(self, img_path):
        img = io.imread(str(img_path))
        img = img[:, :, :3]
        img = cv.resize(img, (self.width, self.height)) / 255.0
        img = (2*img)-1.0
        img_tensor = torch.from_numpy(img).transpose(
            2, 1).transpose(1, 0).float()
        return img_tensor