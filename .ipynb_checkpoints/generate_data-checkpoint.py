import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random
import uuid
import os
import argparse
from huggingface_hub import login

import numpy as np
from transformers import set_seed

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    set_seed(seed)

# =======================
# Argument Parser
# =======================
parser = argparse.ArgumentParser()
parser.add_argument("--model_size", type=int, required=True, help="e.g., 135, 360, 1700")
parser.add_argument("--train_file_count", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--type", type=str, default='pretrained')
parser.add_argument("--load_epoch", type=int, required=True)
parser.add_argument("--train_epoch", type=int, required=True)
parser.add_argument("--num_samples", type=int, default=600000)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--temperature", type=float, default=1.0)
args = parser.parse_args()

# =======================
# Configuration
# =======================
device = "cuda" if torch.cuda.is_available() else "cpu"

if args.seed != 0:
    seed_everything(args.seed)

if args.type == 'distilled':
    model_ckpt_dir = f"./smollm2-{args.model_size}m-{args.type}-train{args.train_file_count}-temp-{args.temperature}-seed{args.seed}-epoch-{args.train_epoch}-ds-new/checkpoint-epoch-{args.load_epoch}"
else:
    model_ckpt_dir = f"./smollm2-{args.model_size}m-{args.type}-train{args.train_file_count}-temp-1.0-seed{args.seed}-epoch-{args.train_epoch}-ds-new/checkpoint-epoch-{args.load_epoch}"
    
tokenizer_hf_path = f"HuggingFaceTB/SmolLM2-{args.model_size}M"
output_dir = f"./smollm2-{args.model_size}m-{args.type}-train{args.train_file_count}-temp-{args.temperature}-seed{args.seed}-epoch-{args.train_epoch}-ds-new-generated-data"
os.makedirs(output_dir, exist_ok=True)

# =======================
# Load model & tokenizer
# =======================

login('HUGGINGFACE_TOKEN')

model = AutoModelForCausalLM.from_pretrained(model_ckpt_dir, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_hf_path)

# Set pad_token to eos_token if not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("BOS:", tokenizer.bos_token, "| PAD:", tokenizer.pad_token, "| EOS:", tokenizer.eos_token)

# =======================
# Generation settings
# =======================
temperature = args.temperature
top_p = 1.0
top_k = model.config.vocab_size
num_return_sequences = 1

inputs = tokenizer(["The"] * args.batch_size, return_tensors="pt").to(device)

# =======================
# Generation loop
# =======================
generated_data = []
for i in range(0, args.num_samples, args.batch_size):
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True,
            max_length=512,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    for j in range(args.batch_size):
        responses = output[j * num_return_sequences : (j + 1) * num_return_sequences]
        decoded_responses = [tokenizer.decode(res, skip_special_tokens=True) for res in responses]
        for response in decoded_responses:
            generated_data.append({"prompt": "The", "response": response})

    # Save every 100,000 samples
    if len(generated_data) >= 100_000:
        random_id = uuid.uuid4().hex
        output_file = os.path.join(output_dir, f"generated_data_part_{random_id}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(generated_data, f, ensure_ascii=False, indent=4)
        generated_data = []

    print(f"Generated: {i + args.batch_size} / {args.num_samples}")

# Final save
if generated_data:
    random_id = uuid.uuid4().hex
    output_file = os.path.join(output_dir, f"generated_data_part_{random_id}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)

print("Data generation complete.")
