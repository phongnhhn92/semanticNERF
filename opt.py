import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/nerf_example_data/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--log_dir', type=str,
                        default='./logs',
                        help='log directory of trained model')
    parser.add_argument('--mode', type=str,
                        default='train',
                        help='mode of model')
    parser.add_argument('--dataset_name', type=str, default='blender',
                        choices=['blender', 'llff', 'carla', 'carlaGVS'],
                        help='which dataset to train/val')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[800, 800],
                        help='resolution (img_w, img_h) of the image')
    parser.add_argument('--num_classes', type=int, default=13)
    parser.add_argument('--num_planes', type=int, default=32)
    parser.add_argument('--num_rays', type=int, default=256,
                        help='Number of random rays in each novel views.')
    parser.add_argument('--N_importance', type=int, default=32,
                        help='number of additional fine samples')
    parser.add_argument('--mpi_encoder_features', type=int, default=96,
                        help='this controls number feature channels at the output of the base encoder-decoder network')

    parser.add_argument('--embedding_size', type=int, default=13,
                        help='when # of semantic classes is large SUN and LTD will be fed with lower dimensoinal embedding of semantics')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='number of uplifted semantic layers')
    parser.add_argument('--stereo_baseline', type=float, default=0.54,
                        help='assumed baseline for converting depth to disparity')
    parser.add_argument('--use_style_encoder',
                        default=False, action='store_true')
    parser.add_argument('--style_feat', type=int, default=256,
                        help='output dimension of the style encoder')
    parser.add_argument('--use_vae', default=True, action='store_true')
    parser.add_argument('--use_disparity_loss',
                        default=False, action='store_true')
    parser.add_argument('--use_Skip', default=False, action='store_true')
    parser.add_argument('--use_style_loss', default=False, action='store_true')
    parser.add_argument('--disparity_weight', default=0.1, type=float,
                        help='for carla=0.1, for other set to 0.5')
    parser.add_argument('--near_plane', type=int, default=1.5,
                        help='nearest plane: 1.5 for carla')
    parser.add_argument('--far_plane', type=int, default=20000,
                        help='far plane: 20000 for carla')

    parser.add_argument('--spheric_poses', default=False, action="store_true",
                        help='whether images are taken in spheric poses (for llff)')

    parser.add_argument('--perturb', type=float, default=1.0,
                        help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0,
                        help='std dev of noise added to regularize sigma')

    parser.add_argument('--batch_size', type=int, default=1024,
                        help='batch size')
    parser.add_argument('--chunk', type=int, default=32*1024,
                        help='chunk size to split the input to avoid OOM')
    parser.add_argument('--num_epochs', type=int, default=16,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--SUN_path', type=str,
                        default='',
                        help='pretrained SUN model')
    # /media/phong/Data2TB/dataset/GVSNet/pre_trained_models/carla/model_epoch_29.pt
    # /media/phong/Data2TB/dataset/GVSNet/pre_trained_models/carla/SUN.pt
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained model weight to load (do not load optimizers, etc)')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer type',
                        choices=['sgd', 'adam', 'radam', 'ranger'])
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        help='scheduler type',
                        choices=['steplr', 'cosine', 'poly'])
    # params for warmup, only applied when optimizer == 'sgd' or 'adam'
    parser.add_argument('--warmup_multiplier', type=float, default=1.0,
                        help='lr is multiplied by this factor after --warmup_epochs')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Gradually warm-up(increasing) learning rate in optimizer')
    ###########################
    #### params for steplr ####
    parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
                        help='scheduler decay step')
    parser.add_argument('--decay_gamma', type=float, default=0.1,
                        help='learning rate decay amount')
    ###########################
    #### params for poly ####
    parser.add_argument('--poly_exp', type=float, default=0.9,
                        help='exponent for polynomial learning rate decay')
    ###########################

    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')

    parser.add_argument('--appearance_feature', type=int, default=32)
    parser.add_argument('--num_upsampling_layers',
                        choices=('normal', 'more', 'most'), default='normal',
                        help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")
    parser.add_argument('--use_instance_mask', action='store_true',
                        help='is paased, instance mask will be assuned to be present')

    # Arguments for SPADE
    parser.add_argument('--d_step_per_g', type=int, default=1,
                        help='num of d updates for each g update')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='Crop to the width of crop_size (after initially scaling the images to load_size.)')
    parser.add_argument('--spade_k_size', type=int, default=3)
    parser.add_argument('--num_D', type=int, default=3)
    parser.add_argument('--output_nc', type=int, default=3)
    parser.add_argument('--n_layers_D', type=int, default=4)

    parser.add_argument('--contain_dontcare_label', action='store_true')
    parser.add_argument('--no_instance', default=True, type=bool)
    parser.add_argument('--norm_G', type=str, default='spectralspadesyncbatch3x3',
                        help='instance normalization or batch normalization')
    parser.add_argument('--norm_D', type=str, default='spectralinstance',
                        help='instance normalization or batch normalization')
    parser.add_argument('--norm_E', type=str, default='spectralinstance',
                        help='instance normalization or batch normalization')
    parser.add_argument('--rgb_loss_coef', type=int, default=1,
                        help='Coefficient of the rgb loss')

    return parser.parse_args()
