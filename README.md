# semanticNERF

#### 17-02-2020:
Starting point: original NERF

#### 17-02-2020:
1. Add feature network
2. Use a single Color network for both coarse and fine NERF
3. Blender (whitebackground) function doesnt work. Dont know why ? If i remove the sigmoid function of the ColorNetwork then it is fine but then the background goes crazy, the content is fine though.
4. Training with LLFF first.

#### Dataset:
Download here: [link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

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
18-02-2020
![alt text](/assets/ideas_18-02-2020.jpg "Logo Title Text 1")