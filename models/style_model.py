import torch
from models.spade import ConvEncoder,SPADEGenerator
from copy import deepcopy
from .spade import KLDLoss

class StyleModel(torch.nn.Module):
    '''
        A wrapper class for predicting MPI and doing rendering
    '''

    def __init__(self, opts):
        super(StyleModel, self).__init__()
        self.opts = opts
        self.encoder = ConvEncoder(opts)
        spade_ltn_opts = deepcopy(opts)
        spade_ltn_opts.__dict__[
            'num_out_channels'] = opts.feats_per_layer
        spade_ltn_opts.__dict__[
            'semantic_nc'] = opts.num_layers * opts.embedding_size
        spade_ltn_opts.__dict__[
            'embedding_size'] = opts.num_layers * opts.embedding_size
        spade_ltn_opts.__dict__[
            'label_nc'] = opts.num_layers * opts.embedding_size
        self.spade_ltn = SPADEGenerator(spade_ltn_opts, no_tanh=True)

        self.get_kld_loss = KLDLoss()

    def _get_scene_encoding(self, input_img):
        if self.opts.use_vae:
            z, mu, logvar = self.encoder(input_img)
            return z, mu, logvar
        else:
            # When we are in test mode unless we explicily want diverse outputs;
            # There is not need to encode the scene and sample from the distribution multiple times
            z, mu, logvar = self.encoder(input_img)
        return z, mu, logvar

    def forward(self, style_img, semantics_nv):

        z, mu, logvar = self._get_scene_encoding(style_img)

        code = z if self.opts.use_vae else mu

        appearance_nv_feats = self.spade_ltn(semantics_nv, code)

        if self.opts.use_vae:
            kld_loss = self.get_kld_loss(mu, logvar) * self.opts.lambda_kld
        else:
            kld_loss = 0.0


        return kld_loss, appearance_nv_feats.data
