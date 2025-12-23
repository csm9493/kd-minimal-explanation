import argparse
import os
import glob
import math
import random
import torch
import wandb
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    set_seed,
)
from transformers.optimization import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict
from pathlib import Path
from transformers.trainer_utils import is_main_process
from transformers.trainer import TRAINING_ARGS_NAME

# === argparse setup ===
parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=int, default=360)
parser.add_argument("--train_file_count", type=int, default=50)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--load_epochs", type=int, default=4)
parser.add_argument("--type", type=str, default='pretrained')
parser.add_argument("--val_file_count", type=int, default=1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
parser.add_argument("--mini_batch_size", type=int, default=64)
parser.add_argument("--temp", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

train_file_count = args.train_file_count
val_file_count = args.val_file_count
seed = args.seed
model_size = args.model_size
epochs = args.epochs
load_epochs = args.load_epochs
type = args.type
temp = args.temp

gradient_accumulation_steps = args.gradient_accumulation_steps
training_mini_batch_size = args.mini_batch_size // args.gradient_accumulation_steps
mini_batch_size = args.mini_batch_size

# === setup ===
model_name = f"HuggingFaceTB/SmolLM2-{model_size}M"
device = "cuda" if torch.cuda.is_available() else "cpu"
samples_per_file = 100000
world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

# === TrainingArguments ===
training_args = TrainingArguments(
    output_dir=f"./smollm2-{model_size}m-{type}-train{train_file_count}-temp-{temp}-seed{seed}-epoch-{load_epochs}-ds-new",
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=5,
    per_device_train_batch_size=training_mini_batch_size,
    max_steps=1,  
    logging_steps=1,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=5e-4,
    optim="adamw_torch",
    report_to="wandb",
    ddp_find_unused_parameters=False,
    logging_dir="./logs",

    fp16=True,  # or bf16=True (if A100)
    deepspeed="deepspeed_config.json",
)

local_rank = training_args.local_rank

# === seed ===
def set_all_seeds(seed: int):
    if is_main_process(local_rank):
        print(f"Setting seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

set_all_seeds(seed)

# === learning steps ===
total_train_samples = train_file_count * samples_per_file
steps_per_epoch = total_train_samples // (mini_batch_size * world_size)
max_steps = steps_per_epoch * epochs
training_args.max_steps = max_steps

if is_main_process(local_rank):
    print(f"Loaded model {model_name}")
    print(f"Using {world_size} GPUs")
    print(f"Max steps: {max_steps}")

# === wandb ===
if is_main_process(local_rank):
    wandb.login(key="WANDB_KEY")
    wandb.init(
        project="smollm2-pretraining",
        name=f"smollm2-{model_size}m-{type}-train{train_file_count}-temp-{temp}-seed{seed}-epoch-{load_epochs}-new-ds",
        resume="allow"
    )

# === model and tokenizer ===
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.apply(lambda module: module.reset_parameters() if hasattr(module, "reset_parameters") else None)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# === tokenize function ===
def tokenize_stream(example: Dict):
    return tokenizer(example["response"], truncation=True, padding="max_length", max_length=512)

# === dataset load ===
def get_full_dataset():

    if type == 'pretrained':
    
        train_files = sorted(glob.glob("smollm2-1.7B-pretrained-generated-data/*.json"))[:train_file_count]

    else:
        train_files = sorted(glob.glob(f"smollm2-360m-pretrained-train{train_file_count}-temp-{temp}-seed{seed}-epoch-{load_epochs}-ds-new-generated-data/*.json"))[:train_file_count]

    if is_main_process(local_rank):
        print(train_files)
    
    dataset = load_dataset("json", data_files=train_files, split="train", streaming=False)
    dataset = dataset.map(tokenize_stream, batched=True)
    return dataset

full_train_dataset = get_full_dataset()
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === WSD scheduler ===
def get_wsd_scheduler(optimizer, warmup_steps, stable_steps, decay_steps):
    total_steps = warmup_steps + stable_steps + decay_steps
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        elif current_step < warmup_steps + stable_steps:
            return 1.0
        elif current_step <= total_steps:
            decay_progress = (current_step - warmup_steps - stable_steps) / max(1, decay_steps)
            return max(0.0, 1.0 - decay_progress)
        else:
            return 0.0
    return lr_lambda


# === Epoch save callback ===
class SaveWithEpochCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        if not is_main_process(args.local_rank):
            return
        trainer = kwargs.get("trainer", None)
        model = trainer.model if trainer else kwargs.get("model", None)
        tokenizer = trainer.tokenizer if trainer else kwargs.get("tokenizer", None)

        epoch_model_dir = os.path.join(args.output_dir, f"checkpoint-epoch-{int(state.epoch)}")
        os.makedirs(epoch_model_dir, exist_ok=True)

        if model is not None:
            model.save_pretrained(epoch_model_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(epoch_model_dir)
        if trainer is not None:
            trainer.state.save_to_json(os.path.join(epoch_model_dir, "trainer_state.json"))
        torch.save(args, os.path.join(epoch_model_dir, TRAINING_ARGS_NAME))
        print(f">>> Model saved at end of epoch {int(state.epoch)} to {epoch_model_dir}")

# === Checkpoint ===
def get_latest_checkpoint(output_dir: str):
    checkpoints = sorted(
        [ckpt for ckpt in Path(output_dir).glob("checkpoint-*") if ckpt.name.replace("checkpoint-", "").isdigit()],
        key=os.path.getmtime,
    )
    return str(checkpoints[-1]) if checkpoints else None

# === CustomTrainer ===
class CustomTrainer(Trainer):
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=training_args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=training_args.weight_decay,
        )
        warmup_steps = int(0.01 * num_training_steps)
        decay_steps = int(0.2 * num_training_steps)
        stable_steps = num_training_steps - warmup_steps - decay_steps
        lr_lambda = get_wsd_scheduler(self.optimizer, warmup_steps, stable_steps, decay_steps)
        self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)

# === Trainer ===
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=full_train_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[SaveWithEpochCallback()],
)

# === start to train ===
checkpoint_path = get_latest_checkpoint(training_args.output_dir)
if checkpoint_path:
    if is_main_process(local_rank):
        print(f"Resuming from checkpoint: {checkpoint_path}")
    trainer.train(resume_from_checkpoint=checkpoint_path)
else:
    if is_main_process(local_rank):
        print("Starting training from scratch...")
    trainer.train()

# === save last model ===
if is_main_process(local_rank):
    model.save_pretrained(f"{training_args.output_dir}/final")
    tokenizer.save_pretrained(f"{training_args.output_dir}/final")
    wandb.finish()
    print("Training complete. Final model saved.")
