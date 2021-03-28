#!/bin/bash
#SBATCH --job-name=PHONG
#SBATCH --account=Project_2001055
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -o train_o.txt
#SBATCH -e train_e.txt
module load gcc/8.3.0 cuda/10.1.168 cudnn cmake


python main.py --dataset_name carla --root_dir /media/phong/Data2TB/dataset/carla/carla/carla_phong_2/Town01/episode_00001/000000 \
               --N_importance 64 --img_wh 800 600 --noise_std 0 --num_epochs 5 --batch_size 1024 --optimizer adam --lr 5e-4 \
               --lr_scheduler steplr --decay_step 2 4 --decay_gamma 0.5 --exp_name exp_carla
