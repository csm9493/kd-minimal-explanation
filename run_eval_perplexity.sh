#!/bin/bash

############################################################

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:4
#SBATCH --constraint="v100|rtx8000|a100"
#SBATCH --job-name=JOB_NAME
#SBATCH --output=hpc_logs/%j.out
#SBATCH --error=hpc_logs/%j.err

# conda init
# source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
# conda activate /scratch/NYU_ID/envs/llms/

############################################################


torchrun --nproc_per_node=4 evaluate_perplexity.py --type 'pretrained' --train_file_count 100 --val_file_count 1 --model_size 360 --seed 0

torchrun --nproc_per_node=4 evaluate_perplexity.py --type 'pretrained' --train_file_count 100 --val_file_count 1 --model_size 135 --seed 0

torchrun --nproc_per_node=4 evaluate_perplexity.py --type 'distilled' --train_file_count 100 --val_file_count 1 --model_size 135 --temp 1.0 --load_epochs 4 --seed 0

torchrun --nproc_per_node=4 evaluate_perplexity.py --type 'distilled' --train_file_count 100 --val_file_count 1 --model_size 135 --temp 0.95 --load_epochs 4 --seed 0

torchrun --nproc_per_node=4 evaluate_perplexity.py --type 'distilled' --train_file_count 100 --val_file_count 1 --model_size 135 --temp 0.875 --load_epochs 4 --seed 0

torchrun --nproc_per_node=4 evaluate_perplexity.py --type 'distilled' --train_file_count 100 --val_file_count 1 --model_size 135 --temp 1.0 --load_epochs 4 --seed 0

