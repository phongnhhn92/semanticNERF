import os
from collections import defaultdict

from pytorch_lightning import LightningModule, Trainer, seed_everything
# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.data import DataLoader

from datasets import dataset_dict
from datasets.carla_utils.utils import SaveSemantics
# losses
from losses import loss_dict
# metrics
from metrics import *
# models
from models.nerf import *
from models.rendering import *
from models.sun_model import SUNModel
from opt import get_opts
# optimizer, scheduler, visualization
from utils import *

#Sample rays
from datasets.ray_utils import *

# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
seed_everything(100)
_DEBUG = True

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams
        self.loss = loss_dict['color'](coef=1)

        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        # NERF model
        self.models = []
        self.nerf_model = NeRF()
        # SUN model
        self.SUN = SUNModel(self.hparams)
        self.models = [self.nerf_model,self.SUN]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, data):
        #TODO: SUN model, NERF training
        loss_dict, semantics_nv, disp_nv = self.SUN(data, mode='training')
        depth_nv = self.hparams.stereo_baseline * (data['k_matrix'][0][0, 0]).view(1, 1) / disp_nv

        # Get rays data
        SB,_,H,W = data['input_seg'].shape
        all_rgb_gt = []
        all_rays = []
        all_semantics = []
        all_depths = []
        semantics_nv = semantics_nv.view(SB, _, -1).permute(0,2,1)
        depth_nv = depth_nv.view(SB, 1, -1).permute(0,2,1)

        for target_rays,target_rays_gt,semantics_nv_b,depth_nv_b \
                in zip(data['target_rays'],data['target_rgb_gt'],semantics_nv,depth_nv):

            #Conver rgb values from 0 to 1
            target_rays_gt = target_rays_gt * 0.5 + 0.5

            #Randomly sample a few rays in the target view.
            pix_inds = torch.randint(0, target_rays.shape[0], (self.hparams.num_rays,))

            rays = target_rays[pix_inds]
            rays_gt = target_rays_gt[pix_inds]
            rays_semantics = semantics_nv_b[pix_inds]
            rays_depth = depth_nv_b[pix_inds]
            all_rgb_gt.append(rays_gt)
            all_rays.append(rays)
            all_semantics.append(rays_semantics)
            all_depths.append(rays_depth)

        all_rgb_gt = torch.stack(all_rgb_gt).view(-1,3)  # (SB * num_rays, 3)
        all_rays = torch.stack(all_rays).view(-1,6)  # (SB * num_rays, 6)
        all_semantics = torch.stack(all_semantics).view(-1,_)
        all_depths = torch.stack(all_depths).view(-1, 1)

        B = all_rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            all_rays[i:i + self.hparams.chunk],
                            all_semantics[i:i + self.hparams.chunk],
                            all_depths[i:i + self.hparams.chunk],
                            self.hparams.near_plane,
                            self.hparams.far_plane,
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk,  # chunk size is effective in val mode
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        return None

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(self.hparams,split='train')
        self.val_dataset = dataset(self.hparams,split='val')


    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=0 if _DEBUG else 4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        results = self(batch)
        # loss = self.loss(results, rgbs, segs.long().squeeze(-1))
        #
        # with torch.no_grad():
        #     typ = 'fine' if 'rgb_fine' in results else 'coarse'
        #     psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        #
        # self.log('lr', get_learning_rate(self.optimizer))
        # self.log('train/loss', loss)
        # self.log('train/psnr', psnr_, prog_bar=True)

        return None

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset,
    #                       shuffle=False,
    #                       num_workers=4,
    #                       batch_size=1,  # validate one image (H*W rays) at a time
    #                       pin_memory=True)
    #
    # def validation_step(self, batch, batch_nb):
    #     rays, rgbs, segs_onehot,segs = batch['rays'], batch['rgbs'], batch['segs_onehot'], batch['segs']
    #     rays = rays.squeeze()  # (H*W, 3)
    #     rgbs = rgbs.squeeze()  # (H*W, 3)
    #     segs = segs.squeeze()  # (H*W)
    #     segs_onehot = segs_onehot.squeeze()  # (H*W, 13)
    #     results = self(rays, segs_onehot)
    #     log = {'val_loss': self.loss(results, rgbs, segs.long())}
    #     typ = 'fine' if 'rgb_fine' in results else 'coarse'
    #
    #     if batch_nb == 0:
    #         W, H = self.hparams.img_wh
    #         img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    #         img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
    #         depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
    #         stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
    #         self.logger.experiment.add_images('val/GT_pred_depth',
    #                                           stack, self.global_step)
    #
    #         # Visualize semantic results
    #         save_semantic = SaveSemantics('carla')
    #         seg_pred = results[f'feature_{typ}'].cpu()
    #         seg_pred = torch.softmax(seg_pred, dim=1)
    #         seg_pred = torch.argmax(seg_pred, dim=1).view(H, W).unsqueeze(0)  # (1,H,W)
    #         seg_pred = torch.from_numpy(save_semantic.to_color(seg_pred)).permute(2, 0, 1)  # (H,W,3)
    #
    #         segs = segs.view(H, W).unsqueeze(0).cpu()
    #         segs = torch.from_numpy(save_semantic.to_color(segs)).permute(2, 0, 1)  # (H,W,3)
    #         stack_segs = torch.stack([segs, seg_pred])
    #         self.logger.experiment.add_images('val/GT_pred_semantics', stack_segs / 255.0, self.global_step)
    #
    #     psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
    #     log['val_psnr'] = psnr_
    #
    #     return log
    #
    # def validation_epoch_end(self, outputs):
    #     mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
    #
    #     self.log('val/loss', mean_loss)
    #     self.log('val/psnr', mean_psnr, prog_bar=True)


def main(hparams):
    system = NeRFSystem(hparams)
    checkpoint_callback = \
        ModelCheckpoint(filename=os.path.join(f'ckpts/{hparams.exp_name}', '{epoch:d}'),
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=5)

    logger = TestTubeLogger(save_dir="logs",
                            name=hparams.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[checkpoint_callback],
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      accelerator='ddp' if hparams.num_gpus > 1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus == 1 else None,
                      deterministic=True)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
