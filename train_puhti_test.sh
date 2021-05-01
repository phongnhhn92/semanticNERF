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

srun python train_GVSNETPlus.py --dataset_name carlaGVS --root_dir /scratch/project_2001055/dataset/GVS \
               --log_dir /scratch/project_2001055/trained_logs/semanticNERF/logs \
               --img_wh 256 256 --num_epochs 1 --batch_size 8 \
               --num_gpus 4 --exp_name exp_carla_GVSNetPlus_test
