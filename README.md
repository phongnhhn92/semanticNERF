# semanticNERF

#### 17-02-2020:
Starting point: original NERF

#### 20-02-2020:
Add carla dataset, check poses to see if we can train NERF on Carla dataset.

#### Dataset:
LLFF,Blender dataset: [link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

Carla dataset: [link](https://drive.google.com/file/d/1ZYIlupT8-Zm7w8G4br2ZoyJfKEEAyEK-/view?ts=6030149b)

#### Prerequisite:
```
Pytorch
Pytorch-Lightning
torch_optimizer
einops
```

#### Training:
```
python main.py --dataset_name llff 
--root_dir /media/phong/Data2TB/dataset/NERF/nerf_llff_data/flower 
--N_importance 64 --img_wh 504 378 --noise_std 0 
--num_epochs 16 --batch_size 1024 
--optimizer adam --lr 5e-4 --lr_scheduler steplr 
--decay_step 2 4 8 --decay_gamma 0.5 
--exp_name exp_flower
```

#### Ideas:
Slides:[link](https://docs.google.com/presentation/d/1s9k5OCkHxywoAk8Ab2kk8J5DApcRCgLtf2DzNNI3nO4/edit#slide=id.gb4f7efcc71_0_64)

18-02-2020: add another feature vector as output of NERF for semantic prediction.
![](/images/img.png "Ideas")
