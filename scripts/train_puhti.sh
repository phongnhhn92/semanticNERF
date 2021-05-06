#!/bin/bash
#SBATCH --job-name=PHONG
#SBATCH --account=Project_2001055
#SBATCH --time=72:00:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH -o full_train_o_carla.txt
#SBATCH -e full_train_e_carla.txt
module load gcc/8.3.0 cuda/10.1.168 cudnn cmake

srun python ../main.py --dataset_name carlaGVS --root_dir /scratch/project_2001055/dataset/GVS \
               --log_dir /scratch/project_2001055/trained_logs/GVSPlus \
               --SUN_path /scratch/project_2001055/dataset/GVS/SUN.pt \
               --N_importance 96 --img_wh 256 256 --noise_std 0.1 --num_epochs 30 --batch_size 4 \
               --num_rays 4096 --optimizer adam --lr 4e-5 --lr_scheduler steplr --decay_step 10 15 25 \
               --decay_gamma 0.5 --num_gpus 4 --exp_name exp_carla_GVSPlus_full
