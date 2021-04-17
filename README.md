# semanticNERF

#### Slide: [link](https://docs.google.com/presentation/d/1j3yNFRC8Yd_XPg7eKPdrRCBsmp2IfE-_BMG9QVAh7EY/edit?usp=sharing)

#### 18-04-2021:
Starting point: Finish training code with simple style encoder (no VAE). Based on the predicted alphas of SUN model to 
sample more dense points. The MLP model is conditioned on (1) semantic class of that ray, (2) style encoded feature F of the entire style image.  

![](/images/GVS_NERF.jpg "Ideas")

#### Dataset:
LLFF,Blender dataset: [link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

Carla dataset: [link](https://drive.google.com/file/d/1ZYIlupT8-Zm7w8G4br2ZoyJfKEEAyEK-/view?ts=6030149b)

#### Prerequisite:
```
Pytorch 1.7.1
Pytorch-Lightning 1.2.0
torch_optimizer 0.1.0 
einops 0.3.0
test-tube
kornia
cv2
```

#### Training:
LLFF dataset
```
python main.py --dataset_name llff 
--root_dir /media/phong/Data2TB/dataset/NERF/nerf_llff_data/flower 
--N_importance 64 --img_wh 504 378 --noise_std 0 
--num_epochs 16 --batch_size 1024 
--optimizer adam --lr 5e-4 --lr_scheduler steplr 
--decay_step 2 4 8 --decay_gamma 0.5 
--exp_name exp_flower
```

CARLA dataset
```
python main.py --dataset_name carla
--root_dir
/media/phong/Data2TB/dataset/carla/carla/carla_phong_2/Town01/episode_00001/000000
--N_importance 64 --img_wh 800 600 --noise_std 0 --num_epochs 5 --batch_size 1024
--optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 2 4  
--decay_gamma 0.5 --exp_name exp_carla
```
