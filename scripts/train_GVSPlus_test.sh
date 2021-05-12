#!/bin/bash
#SBATCH --job-name=PHONG
#SBATCH --account=Project_2001055
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=8G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:4
#SBATCH -o train_o_GVSPlus.txt
#SBATCH -e train_e_GVSPlus.txt
module load gcc/8.3.0 cuda/10.1.168 cudnn cmake

srun python ../train_GVSNETPlus.py --num_gpus 4 --dataset_name carlaGVS --root_dir /scratch/project_2001055/dataset/GVS \
  --log_dir /scratch/project_2001055/trained_logs/GVSPlus \
  --img_wh 256 256 --noise_std 0.1 --num_epochs 1 --batch_size 4 --num_rays 4096 --N_importance 32 \
  --optimizer adam --lr 5e-4 --lr_scheduler steplr --decay_step 2 4 8 12 --decay_gamma 0.5 \
  --exp_name exp_GVSPlus_SemanticRGBLoss
