#!/bin/bash

############################################################

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|a100|v100"
#SBATCH --job-name=JOB_NAME
#SBATCH --output=hpc_logs/%j.ou
#SBATCH --error=hpc_logs/%j.err

# conda init
# source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
# conda activate /scratch/NYU_ID/envs/llms/

############################################################


# Data generation from GT (pretrained Smollm2 1.7B)

python python generate_data_from_gt.py 

# Data generation from p' (360M) and p'' (135M)

python generate_data.py \
  --model_size 360 \
  --train_file_count 100 \
  --seed 0 \
  --load_epoch 4 \
  --type 'pretrained' \
  --temperature 0.9 \
  --num_samples 1000000

python generate_data.py \
  --model_size 135 \
  --train_file_count 100 \
  --seed 0 \
  --load_epoch 1 \
  --type 'pretrained' \
  --num_samples 100000

python generate_data_custom.py \
  --model_size 135 \
  --train_file_count 100 \
  --seed 0 \
  --load_epoch 1 \
  --type 'distilled' \
  --num_samples 100000

