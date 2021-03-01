# semanticNERF

## ERROR:
Wrong layer to fix later.
https://github.com/phongnhhn92/semanticNERF/blob/2365a6e9a4ead702e5f1f7638019b79619e08836/models/nerf.py#L143


#### 23-02-2020:
Starting point: Add one-hot semantic in dataloader. Discuss about the architecture today with Teddy.

#### 26-02-2020:
Dual path semantic NERF network that predicts RGB and semantic maps consistently.
![](/images/dual_path_semanticNERF.jpg "Ideas")

RGB output
![](/images/dual_path_RGB.gif "Ideas")

Semantic output
![](/images/dual_path_semantic.gif "Ideas")

Depth predicted by color density (sigma_c)
![](/images/dual_path_depth.gif "Ideas")

Depth predicted by sematic density (sigma_s)
![](/images/dual_path_depth_seg.gif "Ideas")


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

#### Ideas:
Slides:[link](https://docs.google.com/presentation/d/1s9k5OCkHxywoAk8Ab2kk8J5DApcRCgLtf2DzNNI3nO4/edit#slide=id.gb4f7efcc71_0_64)

18-02-2020: add another feature vector as output of NERF for semantic prediction.
![](/images/img.png "Ideas")

23-02-2020: Add one hot semantic as additional input to NERF. Train significantly faster.
However, the drawback is that we need semantic as input. Can we train a latent feature vector that understand the semantic so that we dont need semantic input but still output the semantic results ?
Later, when we use edited semantic input the output might change robustly.
![](/images/ideas_23-02-2020.jpg "Ideas")
