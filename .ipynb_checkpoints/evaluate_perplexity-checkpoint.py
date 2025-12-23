import os
import glob
import torch
import argparse
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import wandb

# === argument setup ===
parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=int, default=360)
parser.add_argument("--train_file_count", type=int, default=50)
parser.add_argument("--val_file_count", type=int, default=1)
parser.add_argument("--load_epochs", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--temp", type=float, default=1.0)
parser.add_argument("--type", type=str, default='pretrained')
args = parser.parse_args()

train_file_count = args.train_file_count
val_file_count = args.val_file_count
model_size = args.model_size
seed = args.seed
type = args.type
load_epochs = args.load_epochs
temp = args.temp
model_name = f"HuggingFaceTB/SmolLM2-{model_size}M"

# === accelerator init ===
accelerator = Accelerator()
device = accelerator.device

# === seed ===
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# === tokenizer and collator setup ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === validation dataset loading ===
val_files = sorted(glob.glob("smollm2-1.7B-pretrained-validation-data/*.json"))[:1]
print ('val_files : ', val_files)
val_dataset = load_dataset("json", data_files=val_files, split="train")

def tokenize(example):
    return tokenizer(example["response"], truncation=True, padding="max_length", max_length=512)

val_dataset = val_dataset.map(tokenize, batched=True)
val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=data_collator)

# === W&B setup ===
if accelerator.is_main_process:
    wandb.login(key="WANDB_KEY")
    wandb.init(
        project="smollm2-pretraining",
        name=f"smollm2-{model_size}m-{type}-train{train_file_count}-temp-{temp}-seed{seed}-epoch-{load_epochs}-new-ds-validation",
        resume="allow"
    )

# === evaluation function ===
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()

    total_loss = accelerator.gather(torch.tensor(total_loss, device=device)).sum().item()
    total_tokens = accelerator.gather(torch.tensor(total_tokens, device=device)).sum().item()

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return avg_loss, perplexity

if type == 'distilled':
    # smollm2-135m-distilled-train100-temp-0.8-seed0-epoch-4-ds-new
    checkpoints = sorted(Path(f"./smollm2-{model_size}m-{type}-train{train_file_count}-temp-{temp}-seed{seed}-epoch-{load_epochs}-ds-new").glob("checkpoint-epoch-*"), key=os.path.getmtime)
else:
    # smollm2-360m-pretrained-train100-temp-1.0-seed1-epoch-4-ds-new/
    checkpoints = sorted(Path(f"./smollm2-{model_size}m-{type}-train{train_file_count}-temp-{temp}-seed{seed}-epoch-{load_epochs}-ds-new").glob("checkpoint-epoch-*"), key=os.path.getmtime)

for ckpt in checkpoints:
    if accelerator.is_main_process:
        print(f"Evaluating checkpoint: {ckpt}")

    model = AutoModelForCausalLM.from_pretrained(ckpt)
    model.to(device)

    model, val_loader_prepared = accelerator.prepare(model, val_loader)

    val_loss, ppl = evaluate(model, val_loader_prepared)

    if accelerator.is_main_process:
        epoch_num = int(ckpt.name.split("-")[-1])
        print(f"[Epoch {epoch_num}] val_loss = {val_loss:.4f}, perplexity = {ppl:.2f}")
        wandb.log({
            "epoch": epoch_num,
            "val_loss": val_loss,
            "val_perplexity": ppl
        })

if accelerator.is_main_process:
    wandb.finish()
