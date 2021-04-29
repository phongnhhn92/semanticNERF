import os
from collections import defaultdict
from copy import deepcopy

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.data import DataLoader

# datasets
from datasets import dataset_dict
from datasets.carla_utils.utils import SaveSemantics
from datasets.ray_utils import getRandomRays, one_hot_encoding
# losses
from losses import loss_dict
# metrics
from metrics import *
# models
from models.nerf import *
from models.rendering import *
from models.sun_model import SUNModel
from models.spade import ConvEncoder,SPADEGenerator
from models.mpi import ComputeHomography
from models.mpi import AlphaComposition
from models.mpi import ApplyHomography
from models.mpi import Alpha2Disp
from models.mpi import ApplyAssociation
from opt import get_opts
# optimizer, scheduler, visualization
from utils import *

# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
seed_everything(100)
_DEBUG = True

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams
        self.models = {}
        # SUN model
        self.SUN = SUNModel(self.hparams)
        if self.hparams.SUN_path is not None:
            self.SUN.load_state_dict(torch.load(self.hparams.SUN_path))
            self.models['sun'] = self.SUN


        # Scene-style Encoder # from spade
        self.encoder = ConvEncoder(self.hparams)
        # A layered image translator
        spade_ltn_opts = deepcopy(self.hparams)
        spade_ltn_opts.__dict__[
            'num_out_channels'] = self.hparams.num_layers * self.hparams.feats_per_layer
        spade_ltn_opts.__dict__[
            'semantic_nc'] = self.hparams.num_layers * self.hparams.embedding_size
        spade_ltn_opts.__dict__[
            'embedding_size'] = self.hparams.num_layers * self.hparams.embedding_size
        spade_ltn_opts.__dict__[
            'label_nc'] = self.hparams.num_layers * self.hparams.embedding_size
        self.spade_ltn = SPADEGenerator(spade_ltn_opts, no_tanh=True)
        # MPI rendering
        self.compute_homography = ComputeHomography(self.hparams)
        self.alpha_composition = AlphaComposition()
        self.apply_homography = ApplyHomography()
        self.alpha_to_disp = Alpha2Disp(self.hparams)
        self.apply_association = ApplyAssociation(self.hparams.num_layers)

        # RGB loss
        self.loss = loss_dict['color'](coef=self.hparams.rgb_loss_coef)
        # NERF embeddings
        self.embedding_xyz = Embedding(3, 10)
        self.embedding_dir = Embedding(3, 4)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}
        # NERF model
        self.nerf_model = NeRF(in_channels_style=self.hparams.feats_per_layer)
        self.models['nerf'] = self.nerf_model

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, data, training=True):
        # training flag means training SUN network so it always True even it is in the eval mode
        # this training flag will be set to False when we finetune it with NERF network
        results = defaultdict(list)

        # Get style encoding of the scene
        z, mu, logvar = self.encoder(data['style_img'])

        # Get the semantic ,disparity, alpha and appearance feature of the novel view
        layered_sem, mpi_alpha, associations \
            = self.SUN(data, d_loss=self.hparams.use_disparity_loss, mode='inference')

        layered_sem = layered_sem.flatten(1, 2)
        layered_appearance = self.spade_ltn(layered_sem, z=z)
        mpi_appearance = self.apply_association(
            layered_appearance, input_associations=associations)
        # Here we do novel-view synthesis of apearance features
        t_vec, r_mat = data['t_vec'], data['r_mat']
        # Compute planar homography
        h_mats = self.compute_homography(
            kmats=data['k_matrix'], r_mats=r_mat, t_vecs=t_vec)
        mpi_alpha_nv, grid = self.apply_homography(
            h_matrix=h_mats, src_img=mpi_alpha)
        mpi_app_nv, _ = self.apply_homography(
            h_matrix=h_mats, src_img=mpi_appearance, grid=grid)

        # Get rays data
        SB, D, F, H, W = mpi_app_nv.shape
        mpi_app_nv = mpi_app_nv.view(SB,D,F,-1)
        mpi_alpha_nv = mpi_alpha_nv.view(SB,D,-1)

        if training:
            all_rgb_gt, all_rays, all_alphas, all_appearance \
                = getRandomRays(self.hparams, data, mpi_alpha_nv, mpi_app_nv, F)
            chunk = self.hparams.chunk

        print('a')


        results['semantic_nv'] = semantics_nv
        results['disp_iv'] = disp_iv
        results['loss_dict'] = loss_dict

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
                          num_workers=0 if _DEBUG else 8,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        results = self(batch)
        loss = sum([v for k, v in results['loss_dict'].items()])

        self.log('lr', get_learning_rate(self.optimizer))
        if self.hparams.use_disparity_loss:
            self.log('train/disp_loss', results['loss_dict']['disp_loss'])
        self.log('train/semantic_loss', results['loss_dict']['semantics_loss'])

        self.log('train/loss', loss, prog_bar=True)

        return loss

    # def val_dataloader(self):
    #     return DataLoader(self.val_dataset,
    #                       shuffle=False,
    #                       num_workers=0 if _DEBUG else 8,
    #                       batch_size=1,  # validate one image (H*W rays) at a time
    #                       pin_memory=True)
    #
    # def validation_step(self, batch, batch_nb):
    #     results = self(batch)
    #     loss = sum([v for k, v in results['loss_dict'].items()])
    #     log = {'val_loss': loss}
    #
    #     save_semantic = SaveSemantics('carla')
    #     if batch_nb == 0 and _DEBUG is not True:
    #         input_img = batch['input_img'][0].cpu()
    #         input_img = input_img * 0.5 + 0.5
    #
    #         input_seg = torch.argmax(batch['input_seg'], dim=1).cpu()
    #         input_seg = torch.from_numpy(save_semantic.to_color(input_seg)).permute(2, 0, 1)
    #         input_seg = input_seg / 255.0
    #
    #         target_img = batch['target_img'][0].cpu()
    #         target_img = target_img * 0.5 + 0.5
    #
    #         target_seg = torch.argmax(batch['target_seg'], dim=1).cpu()
    #         target_seg = torch.from_numpy(save_semantic.to_color(target_seg)).permute(2, 0, 1)
    #         target_seg = target_seg / 255.0
    #
    #         stack = torch.stack([input_img, input_seg, target_img, target_seg])
    #
    #         pred_seg = torch.argmax(results['semantic_nv'], dim=1).cpu()
    #         pred_seg = torch.from_numpy(save_semantic.to_color(pred_seg)).permute(2, 0, 1)
    #         pred_seg = pred_seg / 255.0
    #
    #         pred_disp = save_depth(results['disp_iv'].squeeze().cpu())
    #
    #         stack_pred = torch.stack([pred_seg, pred_disp])
    #
    #
    #         self.logger.experiment.add_images('val/rgb_sem_INPUT-rgb_sem_TARGET',
    #                                               stack, self.global_step)
    #         self.logger.experiment.add_images('val/predictions',
    #                                               stack_pred, self.global_step)
    #
    #     return log
    #
    # def validation_epoch_end(self, outputs):
    #     mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #
    #     self.log('val/loss', mean_loss,prog_bar=True)


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
                            debug=_DEBUG,
                            create_git_tag=False,
                            log_graph=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[checkpoint_callback],
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=1000 if hparams.num_gpus > 1 else 1,
                      num_nodes = 1,
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
