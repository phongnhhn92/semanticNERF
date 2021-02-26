import glob, os
from PIL import Image
import numpy as np
import imageio

PATH = '/media/phong/Data2TB/Paper4_backup/26-02-2020_NERF_semantic_consitent/NERF_semantic/results/carla/images'

imgs = []
for n,file in enumerate(glob.glob(os.path.join(PATH,"*.png"))):
    img =  np.asarray(Image.open(file))
    imgs += [img]

imageio.mimsave('out.gif',imgs,fps=30)

