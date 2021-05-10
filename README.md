# semanticNERF

#### Slide: [link](https://docs.google.com/presentation/d/1j3yNFRC8Yd_XPg7eKPdrRCBsmp2IfE-_BMG9QVAh7EY/edit?usp=sharing)

#### 18-04-2021:
Starting point: Finish training code with simple style encoder (no VAE). Based on the predicted alphas of SUN model to 
sample more dense points. The MLP model is conditioned on (1) semantic class of that ray, (2) style encoded feature F of the entire style image.
#### 09-05-2021:
-Pretrained SUN

-Resnet 18 encoder + Upsample blocks which include SPADE Resblock

-Homography warping to get the MPI appearance of the novel view

-Concat output of encoder (MPI appearance) with MPI semantic 
-> input to NERF

#### 10-05-2021:
Predicted MPI alpha is feed to an AlphaMLP to sample a mapping from a coarse-to-dense appearance MPI
-> Sample more points along the ray, rather than just 32 points of SUN network

![](/images/GVS_NERF.jpg "Ideas")

#### Dataset:
LLFF,Blender dataset: [link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

Carla dataset: [link](https://drive.google.com/file/d/1f7zPW9U3BOOb9aZMg5YOiL_r5MFNIq2f/view)

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

CARLA dataset
```
python main.py --dataset_name carla
--dataset_name carlaGVS 
--root_dir /media/phong/Data2TB/dataset/carlaGVSNet 
--N_importance 64 --img_wh 256 256 --noise_std 0.1 --num_epochs 16 
--batch_size 3 --num_rays 256 --N_importance 96 --optimizer adam 
--lr 4e-5 --lr_scheduler steplr --decay_step 2 4 8 12 
--decay_gamma 0.5 --exp_name exp_carla_GVS
```
