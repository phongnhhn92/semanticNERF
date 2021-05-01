import os
# pytorch-lightning
from itertools import chain

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from torch.utils.data import DataLoader

# datasets
from datasets import dataset_dict
from datasets.carla_utils.utils import SaveSemantics
from models.gvsNetPlus import GVSNet
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
        self.models = {}
        # SUN model
        self.model = GVSNet(self.hparams)

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, data, mode):
        gen_loss_dict = self.model(data, mode)
        return gen_loss_dict

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        self.train_dataset = dataset(self.hparams, split='train')
        self.val_dataset = dataset(self.hparams, split='val')

    def configure_optimizers(self):
        gen_param_list = []
        gen_param_list.extend([self.model.adn.parameters()])
        gen_param_list.extend([self.model.sun.parameters()])
        gen_param_list.extend([self.model.encoder.parameters()])
        gen_param_list.extend([self.model.spade_ltn.parameters()])

        self.gen_optimizer = torch.optim.Adam(
            chain(*gen_param_list), lr=self.hparams.gen_lr, betas=(0.9, 0.999))

        self.disc_optimizer = torch.optim.Adam(self.model.discriminator.parameters(),
                                               lr=self.hparams.disc_lr, betas=(0, 0.999))

        self.disc_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.disc_optimizer, lr_lambda=lr_func(self.hparams.num_epochs))
        self.gen_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.gen_optimizer, lr_lambda=lr_func(self.hparams.num_epochs))

        return [self.gen_optimizer, self.disc_optimizer], [self.gen_scheduler, self.disc_scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=0 if _DEBUG else 8,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def training_step(self, batch, batch_nb, optimizer_idx):
        self.log('train/lr_g', get_learning_rate(self.gen_optimizer))
        self.log('train/lr_d', get_learning_rate(self.disc_optimizer))
        # Generator
        if optimizer_idx == 0:
            gen_results = self(batch, mode='generator')
            g_loss = sum(
                [v for k, v in gen_results.items() if not v is None])
            self.log('train/g_loss', g_loss, prog_bar=True, on_step=True)

            if batch_nb % 1000 == 0:
                real, fake = self.model.real[0].cpu(), self.model.fake[0].cpu()
                real = real * 0.5 + 0.5
                fake = fake * 0.5 + 0.5
                stack = torch.stack([real, fake, ])
                self.logger.experiment.add_images('train/real_fake', stack, self.global_step)

            return g_loss
        # Discriminator
        if optimizer_idx == 1:
            dis_results = self(batch, mode='discriminator')
            d_loss = sum([v for k, v in dis_results.items() if not v is None])
            self.log('train/d_loss', d_loss, prog_bar=True, on_step=True)
            return d_loss

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=0 if _DEBUG else 8,
                          batch_size=1,  # validate one image (H*W rays) at a time
                          pin_memory=True)

    def validation_step(self, batch, batch_nb):
        results = self(batch, mode='generator')
        loss = sum(
            [v for k, v in results.items() if not v is None])
        log = {'val_loss': loss}

        save_semantic = SaveSemantics('carla')
        if batch_nb == 0 and _DEBUG is not True:
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

            pred_seg = torch.argmax(self.model.sem_nv[0], dim=0).cpu()
            pred_seg = torch.from_numpy(save_semantic.to_color(pred_seg)).permute(2, 0, 1)
            pred_seg = pred_seg / 255.0

            pred_disp = save_depth(self.model.disp_iv.squeeze().cpu())

            pred_rgb = self.model.fake[0].cpu()
            pred_rgb = pred_rgb * 0.5 + 0.5

            stack_pred = torch.stack([pred_rgb, pred_seg, pred_disp])

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
                            debug=_DEBUG,
                            create_git_tag=False,
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
