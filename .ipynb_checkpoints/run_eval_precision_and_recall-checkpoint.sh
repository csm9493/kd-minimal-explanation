#!/bin/bash

#############################################################

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|v100"
#SBATCH --job-name=KD_PR_seed_0_new
#SBATCH --output=hpc_logs/%j.out
#SBATCH --error=hpc_logs/%j.err

# conda init
# source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh
# conda activate /scratch/NYU_ID/envs/llms/

#############################################################



# PR: 1.7B and pretrained 360M
torchrun --nproc_per_node=1 --master_port=29604 evaluate_precision_recall.py \
  --seed 0 \
  --pstar_model_dir HuggingFaceTB/SmolLM2-1.7B \
  --pprime_model_dir ./smollm2-360m-pretrained-train100-seed0-ds-new/checkpoint-epoch-4 \
  --pstar_data_dir ./smollm2-1.7B-pretrained-validation-data \
  --pprime_data_dir ./smollm2-360m-pretrained-train100-seed0-ds-new-validation-data \
  --sample_size 100000 

# PR: 1.7B and pretrained 135M
torchrun --nproc_per_node=1 --master_port=29605 evaluate_precision_recall.py \
  --seed 0 \
  --pstar_model_dir HuggingFaceTB/SmolLM2-1.7B \
  --pprime_model_dir ./smollm2-135m-pretrained-train100-seed0-ds-new/checkpoint-epoch-1 \
  --pstar_data_dir  ./smollm2-1.7B-pretrained-validation-data \
  --pprime_data_dir ./smollm2-135m-pretrained-train100-seed0-ds-new-validation-data \
  --sample_size 100000 

# PR: 1.7B and distilled 135M temp 1.0
torchrun --nproc_per_node=1 --master_port=29600 evaluate_precision_recall.py \
  --seed 0 \
  --pstar_model_dir HuggingFaceTB/SmolLM2-1.7B \
  --pprime_model_dir ./smollm2-135m-distilled-train100-temp-1.0-seed0-epoch-4-ds-new/checkpoint-epoch-1 \
  --pstar_data_dir  ./smollm2-1.7B-pretrained-validation-data \
  --pprime_data_dir ./smollm2-135m-distilled-train100-temp-1.0-seed0-epoch-4-ds-new-validation-data \
  --sample_size 100000 

# PR: 1.7B and distilled 135M temp 0.95
torchrun --nproc_per_node=1 --master_port=29603 evaluate_precision_recall.py \
  --seed 0 \
  --pstar_model_dir HuggingFaceTB/SmolLM2-1.7B \
  --pprime_model_dir ./smollm2-135m-distilled-train100-temp-0.95-seed0-epoch-4-ds-new/checkpoint-epoch-1 \
  --pstar_data_dir  ./smollm2-1.7B-pretrained-validation-data \
  --pprime_data_dir ./smollm2-135m-distilled-train100-temp-0.95-seed0-epoch-4-ds-new-validation-data \
  --sample_size 100000 

# PR: 1.7B and distilled 135M temp 0.875
torchrun --nproc_per_node=1 --master_port=29601 evaluate_precision_recall.py \
  --seed 0 \
  --pstar_model_dir HuggingFaceTB/SmolLM2-1.7B \
  --pprime_model_dir ./smollm2-135m-distilled-train100-temp-0.875-seed0-epoch-4-ds-new/checkpoint-epoch-1 \
  --pstar_data_dir  ./smollm2-1.7B-pretrained-validation-data \
  --pprime_data_dir ./smollm2-135m-distilled-train100-temp-0.875-seed0-epoch-4-ds-new-validation-data \
  --sample_size 100000 

# PR: 1.7B and distilled 135M temp 0.8
torchrun --nproc_per_node=1 --master_port=29602 evaluate_precision_recall.py \
  --seed 0 \
  --pstar_model_dir HuggingFaceTB/SmolLM2-1.7B \
  --pprime_model_dir ./smollm2-135m-distilled-train100-temp-0.8-seed0-epoch-4-ds-new/checkpoint-epoch-1 \
  --pstar_data_dir  ./smollm2-1.7B-pretrained-validation-data \
  --pprime_data_dir ./smollm2-135m-distilled-train100-temp-0.8-seed0-epoch-4-ds-new-validation-data \
  --sample_size 100000 

  

