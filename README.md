# MiP-NERF

#### 05-04-2020:
Starting point: original MiP-NERF

#### Dataset:
LLFF,Blender dataset: [link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

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
Blender dataset (Mip-NERF is optimized for this dataset, not LLFF)
```
python main.py --dataset_name blender 
--root_dir /media/phong/Data2TB/dataset/NERF/nerf_synthetic/nerf_synthetic/lego 
--N_importance 128 --img_wh 800 800 --noise_std 0 --num_epochs 16 --batch_size 4096 
--optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 2 4 --decay_gamma 0.5 
--exp_name exp_lego
```

LLFF dataset (not tested yet)
```
python main.py --dataset_name llff 
--root_dir /media/phong/Data2TB/dataset/NERF/nerf_llff_data/flower 
--N_importance 64 --img_wh 504 378 --noise_std 0 
--num_epochs 16 --batch_size 1024 
--optimizer adam --lr 5e-4 --lr_scheduler steplr 
--decay_step 2 4 8 --decay_gamma 0.5 
--exp_name exp_flower
```

