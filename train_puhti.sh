#!/bin/bash
#SBATCH --job-name=PHONG
#SBATCH --account=Project_2001055
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH -o train_o_carla.txt
#SBATCH -e train_e_carla.txt
module load gcc/8.3.0 cuda/10.1.168 cudnn cmake


python main.py --dataset_name carlaGVS --root_dir /path \
               --N_importance 64 --img_wh 256 256 --noise_std 0.1 --num_epochs 16 --batch_size 4 \
               --num_rays 1024 --optimizer adam --lr 4e-5 --lr_scheduler steplr --decay_step 2 4 8 12 \
               --decay_gamma 0.5 --exp_name exp_carla_GVS
