import os
from collections import defaultdict

from einops import repeat
# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.data import DataLoader

# datasets
from datasets import dataset_dict
from datasets.carla_utils.utils import SaveSemantics
from datasets.ray_utils import getRandomRays
# losses
from losses import loss_dict
# metrics
from metrics import *
from models.conv_network import BaseEncoder
# models
from models.nerf import *
from models.rendering import *
from models.sun_model import SUNModel
from opt import get_opts
# optimizer, scheduler, visualization
from utils import *

# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
seed_everything(100)
_DEBUG = False


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
        self.nerf_model = NeRF()
        # SUN model
        self.SUN = SUNModel(self.hparams)
        # Style encoder
        self.encoder = BaseEncoder(in_chans=3, output_feats=self.hparams.style_feat)
        self.models = {'nerf': self.nerf_model, 'sun': self.SUN, 'enoder': self.encoder}

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, data, training=True):
        loss_dict, semantics_nv, disp_nv, alpha_nv = self.SUN(data, d_loss=self.hparams.use_disparity_loss)
        style_code = self.encoder(data['input_img'])
        # Get rays data
        SB, _, H, W = data['input_seg'].shape
        semantics_nv = semantics_nv.view(SB, _, -1).permute(0, 2, 1)
        alpha_nv = alpha_nv.view(SB, self.hparams.num_planes, -1).permute(0, 2, 1)

        if training:
            all_rgb_gt, all_rays, all_semantics, all_alphas, all_styles \
                = getRandomRays(self.hparams, data, semantics_nv, alpha_nv, style_code)
            chunk = self.hparams.chunk
        else:
            assert SB == 1, 'Wrong eval batch size !'
            all_rgb_gt = data['target_rgb_gt'].squeeze(0)
            all_rays = data['target_rays'].squeeze(0)
            all_semantics = semantics_nv.squeeze(0)
            all_alphas = alpha_nv.squeeze(0)
            all_styles = repeat(style_code, '1 n1 -> r n1', r=all_rgb_gt.shape[0])
            chunk = self.hparams.chunk // 8

        B = all_rays.shape[0]
        results = defaultdict(list)

        for i in range(0, B, chunk):
            rendered_ray_chunks = \
                render_rays(self.nerf_model,
                            self.embeddings,
                            all_rays[i:i + chunk],
                            all_semantics[i:i + chunk],
                            all_alphas[i:i + chunk],
                            all_styles[i:i + chunk],
                            self.hparams.near_plane,
                            self.hparams.far_plane,
                            self.hparams.num_planes,
                            self.hparams.N_importance,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.chunk,  # chunk size is effective in val mode
                            )

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)

        loss_dict['rgb_loss'] = self.loss(results, all_rgb_gt)
        results['semantic_nv'] = semantics_nv
        results['disp_nv'] = disp_nv
        results['loss_dict'] = loss_dict

        psnr_ = psnr(results[f'rgb'], all_rgb_gt)
        results['psnr'] = psnr_
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(self.hparams, split='train')
        self.val_dataset = dataset(self.hparams, split='val')

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
        loss = sum([v for k, v in results['loss_dict'].items()])

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/rgb_loss', results['loss_dict']['rgb_loss'])
        if self.hparams.use_disparity_loss:
            self.log('train/disp_loss', results['loss_dict']['disp_loss'])
        self.log('train/semantic_loss', results['loss_dict']['semantics_loss'])

        self.log('train/loss', loss)
        self.log('train/psnr', results['psnr'], prog_bar=True)

        return loss

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=0 if _DEBUG else 4,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)

    def validation_step(self, batch, batch_nb):
        results = self(batch, training=False)
        loss = sum([v for k, v in results['loss_dict'].items()])
        log = {'val_loss': loss}

        save_semantic = SaveSemantics('carla')
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            input_img = batch['input_img'][0].cpu()
            input_img = input_img * 0.5 + 0.5

            input_seg = torch.argmax(batch['input_seg'], dim=1).cpu()
            input_seg = torch.from_numpy(save_semantic.to_color(input_seg)).permute(2, 0, 1)
            input_seg = input_seg / 255.0

            target_img = batch['target_img'][0].cpu()
            target_img = target_img * 0.5 + 0.5

            target_seg = torch.argmax(batch['target_seg'], dim=1).cpu()
            target_seg = torch.from_numpy(save_semantic.to_color(target_seg)).permute(2, 0, 1)
            target_seg = target_seg / 255.0

            stack = torch.stack([input_img, input_seg, target_img, target_seg])
            self.logger.experiment.add_images('val/rgb_sem_INPUT-rgb_sem_TARGET',
                                              stack, self.global_step)

            pred_seg = results['semantic_nv'].permute(0, 2, 1).view(1, self.hparams.num_classes, H, W).cpu()
            pred_seg = torch.argmax(pred_seg, dim=1)
            pred_seg = torch.from_numpy(save_semantic.to_color(pred_seg)).permute(2, 0, 1)
            pred_seg = pred_seg / 255.0

            pred_rgb = results['rgb'].permute(1, 0).view(3, H, W).cpu()
            pred_disp = visualize_depth(results['disp_nv'].squeeze().cpu())
            pred_depth = visualize_depth(results['depth'].view(H, W).cpu())

            stack_pred = torch.stack([pred_rgb, pred_seg, pred_disp, pred_depth])
            self.logger.experiment.add_images('val/predictions',
                                              stack_pred, self.global_step)

        log['val_psnr'] = results['psnr']

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)


def main(hparams):
    system = NeRFSystem(hparams)
    checkpoint_callback = \
        ModelCheckpoint(dirpath=os.path.join(hparams.log_dir,f'ckpts/{hparams.exp_name}'),
                        filename='{epoch}-{val_loss:.2f}',
                        monitor='val/psnr',
                        mode='max',
                        save_top_k=5)

    logger = TestTubeLogger(save_dir=hparams.log_dir,
                            name=hparams.exp_name,
                            debug=_DEBUG,
                            create_git_tag=False,
                            log_graph=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[checkpoint_callback],
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1000 if hparams.num_gpus > 1 else 1,
                      gpus=hparams.num_gpus,
                      accelerator='ddp' if hparams.num_gpus > 1 else None,
                      sync_batchnorm=True if hparams.num_gpus > 1 else False,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus == 1 else None,
                      deterministic=True)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
