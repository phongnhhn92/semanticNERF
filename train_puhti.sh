#!/bin/bash
#SBATCH --job-name=PHONG
#SBATCH --account=Project_2001055
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -o train_o_1.txt
#SBATCH -e train_e_1.txt
module load gcc/8.3.0 cuda/10.1.168 cudnn cmake

python main.py --dataset_name blender --root_dir /media/phong/Data2TB/dataset/NERF/nerf_synthetic/nerf_synthetic/lego \
--N_importance 128 --img_wh 800 800 --noise_std 0 --num_epochs 16 --batch_size 4096 --optimizer adam --lr 5e-4 \
--lr_scheduler steplr --decay_step 2 4 8 --decay_gamma 0.5 --exp_name exp_lego