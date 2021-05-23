import os
# pytorch-lightning
from collections import defaultdict

from einops import rearrange
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
from models.alpha_MLP import Alpha_MLP
from models.backboned_unet.unet import Unet
# models
from models.mpi import ApplyAssociation, ComputeHomography, ApplyHomography
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
        self.loss = loss_dict['color'](coef=self.hparams.rgb_loss_coef)

        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}
        self.apply_association = ApplyAssociation(self.hparams.num_layers)
        self.compute_homography = ComputeHomography(self.hparams)
        self.apply_homography = ApplyHomography()

        self.feature_models = {}
        # NERF model
        self.nerf_model = NeRF(in_channels_style=self.hparams.appearance_feature + self.hparams.embedding_size)
        # Alpha MLP
        self.alpha = Alpha_MLP(in_channels=self.hparams.num_planes,
                               out_channels=self.hparams.num_planes * (
                                       self.hparams.num_planes + self.hparams.N_importance))
        self.mlp_model = {'nerf': self.nerf_model, 'alpha': self.alpha}

        # SUN model
        self.SUN = SUNModel(self.hparams)
        if self.hparams.SUN_path != '':
            self.SUN.load_state_dict(torch.load(self.hparams.SUN_path))
            self.SUN.eval()
        else:
            self.feature_models['sun'] = self.SUN

        # Original weight use SyncBatchNorm, replace them with Batchnorm
        self.SUN = convert_model(self.SUN)

        # Encoder
        self.encoder = Unet(self.hparams, backbone_name='resnet18',
                            pretrained=True,
                            encoder_freeze=True,
                            out_channels=self.hparams.num_layers * self.hparams.appearance_feature,
                            parametric_upsampling=False)
        self.feature_models['encoder'] = self.encoder
        print('Init models !!!')



    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, data, training=False):
        # Get the semantic ,disparity, alpha and appearance feature of the novel view
        if self.hparams.SUN_path != '':
            with torch.no_grad():
                _, seg_mul_layer, grid, associations, semantics_nv, mpi_semantics_nv, disp_nv, mpi_alpha_nv \
                    = self.SUN(data)
        else:
            sun_loss,seg_mul_layer, grid, associations, semantics_nv, mpi_semantics_nv, disp_nv, mpi_alpha_nv \
                = self.SUN(data)
        seg_mul_layer = seg_mul_layer.flatten(1, 2)

        # Encoder
        B, S, H, W = data['input_seg'].shape
        layered_appearance = self.encoder(data['style_img'], seg_mul_layer)
        layered_appearance = layered_appearance.view(
            B, self.hparams.num_layers, self.hparams.appearance_feature, H, W)
        mpi_appearance = self.apply_association(
            layered_appearance, input_associations=associations)

        # Here we do novel-view synthesis of apearance features
        t_vec, r_mat = data['t_vec'], data['r_mat']
        # Compute planar homography
        h_mats = self.compute_homography(
            kmats=data['k_matrix'], r_mats=r_mat, t_vecs=t_vec)
        mpi_appearance_nv, _ = self.apply_homography(
            h_matrix=h_mats, src_img=mpi_appearance, grid=grid)

        SB, D, F, H, W = mpi_appearance_nv.shape
        mpi_appearance_nv = rearrange(mpi_appearance_nv, 'b d f h w -> b (h w) d f')
        mpi_semantics_nv = rearrange(mpi_semantics_nv, 'b d f h w -> b (h w) d f')
        mpi_alpha_nv = rearrange(mpi_alpha_nv.squeeze(2), 'b d h w -> b (h w) d')

        if training:
            all_rgb_gt, all_rays, all_alphas, all_appearance, all_semantic \
                = getRandomRays(self.hparams, data, mpi_alpha_nv, mpi_appearance_nv, mpi_semantics_nv, F)
            chunk = self.hparams.chunk
        else:
            assert SB == 1, 'Wrong eval batch size !'
            all_rgb_gt = data['target_rgb_gt']
            all_rays = data['target_rays']
            all_appearance = mpi_appearance_nv
            all_alphas = mpi_alpha_nv
            all_semantic = mpi_semantics_nv
            chunk = self.hparams.chunk // 16

        final_results = {}
        # Concat feature and semantic maps
        all_appearance = torch.cat([all_appearance, all_semantic], dim=-1)
        for b in range(SB):
            results = defaultdict(list)
            R = all_rays[b].shape[0]
            # Conditional NERF MLP network
            for i in range(0, R, chunk):
                rendered_ray_chunks = \
                    render_rays(self.mlp_model,
                                self.embeddings,
                                all_rays[b][i:i + chunk],
                                all_alphas[b][i:i + chunk],
                                all_appearance[b][i:i + chunk],
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

            if b == 0:
                for k, v in results.items():
                    final_results[k] = results[k]
            else:
                for k, v in results.items():
                    final_results[k] = torch.cat([final_results[k], results[k]], dim=0)
        for k, v in final_results.items():
            if training:
                assert final_results[k].shape[0] == SB * self.hparams.num_rays, 'Error reshaping !'
                final_results[k] = final_results[k].view(SB, self.hparams.num_rays, -1)
            else:
                final_results[k] = final_results[k].unsqueeze(0)

        loss = {}
        loss['rgb_loss'] = self.loss(final_results, all_rgb_gt)
        if self.hparams.SUN_path == '':
            loss['semantic_loss'] = sun_loss['semantics_loss']
            loss['disp_loss'] = sun_loss['disp_loss']
        final_results['semantic_nv'] = semantics_nv
        final_results['disp_nv'] = disp_nv
        final_results['loss_dict'] = loss
        psnr_ = psnr(final_results[f'rgb'], all_rgb_gt)
        final_results['psnr'] = psnr_
        return final_results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        self.train_dataset = dataset(self.hparams, split='train')
        self.val_dataset = dataset(self.hparams, split='val')

    def configure_optimizers(self):
        self.optimizer = get_optimizer2(self.hparams, self.feature_models, self.mlp_model)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=0 if _DEBUG else 8,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        self.log('train/lr', get_learning_rate(self.optimizer))

        results = self(batch, training=True)
        loss = sum(
            [v for k, v in results['loss_dict'].items()])
        self.log('train/rgb_loss', results['loss_dict']['rgb_loss'])
        if self.hparams.SUN_path == '':
            self.log('train/semantic_loss', results['loss_dict']['semantic_loss'])
            self.log('train/disp_loss', results['loss_dict']['disp_loss'])
        self.log('train/loss', loss)
        self.log('train/psnr', results['psnr'], prog_bar=True)
        return loss

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=0 if _DEBUG else 8,
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

            input_seg = torch.argmax(batch['input_seg'][0], dim=0).cpu()
            input_seg = torch.from_numpy(save_semantic.to_color(input_seg)).permute(2, 0, 1)
            input_seg = input_seg / 255.0
            # from torchvision.utils import save_image
            # save_image(input_seg, 'img1.png')

            target_img = batch['target_img'][0].cpu()
            target_img = target_img * 0.5 + 0.5

            target_seg = torch.argmax(batch['target_seg'][0], dim=0).cpu()
            target_seg = torch.from_numpy(save_semantic.to_color(target_seg)).permute(2, 0, 1)
            target_seg = target_seg / 255.0

            stack = torch.stack([input_img, input_seg, target_img, target_seg])

            pred_seg = torch.argmax(results['semantic_nv'].squeeze(), dim=0).cpu()
            pred_seg = torch.from_numpy(save_semantic.to_color(pred_seg)).permute(2, 0, 1)
            pred_seg = pred_seg / 255.0

            pred_disp = save_depth(results['disp_nv'].squeeze().cpu())
            baseline = self.hparams.stereo_baseline
            fx = 128.0
            pred_depth_cvt = baseline * fx / results['depth']
            pred_depth = save_depth(pred_depth_cvt.squeeze().view(H, W).cpu())
            pred_rgb = results['rgb'].squeeze().permute(1, 0).view(3, H, W).cpu()

            stack_pred = torch.stack([pred_rgb, pred_seg, pred_disp, pred_depth])

            self.logger.experiment.add_images('val/rgb_sem_INPUT-rgb_sem_TARGET',
                                              stack, self.global_step)
            self.logger.experiment.add_images('val/predictions',
                                              stack_pred, self.global_step)

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        self.log('val/loss', mean_loss, prog_bar=True)


def main(hparams):
    system = NeRFSystem(hparams)
    checkpoint_callback = \
        ModelCheckpoint(dirpath=os.path.join(hparams.log_dir, f'ckpts/{hparams.exp_name}'),
                        filename='{epoch}-{val_loss:.2f}',
                        monitor='val/loss',
                        mode='max',
                        save_top_k=5)

    logger = TestTubeLogger(save_dir=hparams.log_dir,
                            name=hparams.exp_name,
                            debug=False,
                            create_git_tag=True,
                            log_graph=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[checkpoint_callback],
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1000 if hparams.num_gpus > 1 else 1,
                      num_nodes=1,
                      gpus=hparams.num_gpus,
                      accelerator='ddp' if hparams.num_gpus > 1 else None,
                      sync_batchnorm=True if hparams.num_gpus > 1 else False,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus == 1 else None,
                      deterministic=False)

    trainer.fit(system)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)
