#!/bin/bash

#################################################################

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:4
#SBATCH --constraint="v100|rtx8000"
#SBATCH --job-name=JOB_NAME
#SBATCH --output=hpc_logs/%j.out
#SBATCH --error=hpc_logs/%j.err

# conda init
# source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
# conda activate /scratch/NYU_ID/envs/llms/

#################################################################


# Train p' (360M)

torchrun --nproc_per_node=4 train_model.py --model_size 360 --train_file_count 100 --val_file_count 1 --mini_batch_size 64 --gradient_accumulation_steps 4 --epochs 5 --seed 0 --type 'pretrained'

# Train p'' (135M)

torchrun --nproc_per_node=4 train_model.py --model_size 135 --train_file_count 100 --val_file_count 1 --mini_batch_size 64 --gradient_accumulation_steps 2 --epochs 5 --seed 0 --type 'pretrained'

# Train p'' (135M) via KD (sampled data from p' [temp = 0.8])

torchrun --nproc_per_node=4 train_model.py --model_size 135 --train_file_count 100 --val_file_count 1 --mini_batch_size 64 --gradient_accumulation_steps 2 --epochs 5 --seed 0 --type 'distilled' --temp 0.8

# Train p'' (135M) via KD (sampled data from p' [temp = 0.875])

torchrun --nproc_per_node=4 train_model.py --model_size 135 --train_file_count 100 --val_file_count 1 --mini_batch_size 64 --gradient_accumulation_steps 2 --epochs 5 --seed 0 --type 'distilled' --temp 0.875

# Train p'' (135M) via KD (sampled data from p' [temp = 0.95])

torchrun --nproc_per_node=4 train_model.py --model_size 135 --train_file_count 100 --val_file_count 1 --mini_batch_size 64 --gradient_accumulation_steps 2 --epochs 5 --seed 0 --type 'distilled' --temp 0.95

# Train p'' (135M) via KD (sampled data from p' [temp = 1.0])

torchrun --nproc_per_node=4 train_model.py --model_size 135 --train_file_count 100 --val_file_count 1 --mini_batch_size 64 --gradient_accumulation_steps 2 --epochs 5 --seed 0 --type 'distilled' --temp 1.0

