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

        self.nerf_coarse = NeRF()
        self.models = {'coarse': self.nerf_coarse}
        load_ckpt(self.nerf_coarse, hparams.weight_path, 'nerf_coarse')

        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models['fine'] = self.nerf_fine
            load_ckpt(self.nerf_fine, hparams.weight_path, 'nerf_fine')

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, data):

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
