#!/bin/bash
#SBATCH --job-name=PHONG
#SBATCH --account=Project_2001055
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:4
#SBATCH -o train_o_carla.txt
#SBATCH -e train_e_carla.txt
module load gcc/8.3.0 cuda/10.1.168 cudnn cmake

srun python ../train_SUN.py --dataset_name carlaGVS --root_dir /media/phong/Data2TB/dataset/carlaGVSNet \
        --img_wh 256 256 --noise_std 0.1 --num_epochs 30 --batch_size 16  --num_rays 32 --N_importance 96 \
        --optimizer adam --lr 0.0004 --lr_scheduler steplr  --decay_step 10 20  --decay_gamma 0.5 \
        --use_disparity_loss --exp_name exp_carla_GVS_SUN
