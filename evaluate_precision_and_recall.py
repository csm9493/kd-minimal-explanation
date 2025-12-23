import argparse
import torch
import numpy as np
import wandb
import os
import glob
import json
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from accelerate import Accelerator
from math import exp

# === argparse setup ===
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sample_size", type=int, default=100000)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--pstar_model_dir", type=str, required=True)
parser.add_argument("--pprime_model_dir", type=str, required=True)
parser.add_argument("--pstar_data_dir", type=str, default="A")
parser.add_argument("--pprime_data_dir", type=str, default="B")
args = parser.parse_args()

# === setup ===
seed = args.seed
sample_size = args.sample_size
batch_size = args.batch_size
max_length = args.max_length
pstar_model_dir = args.pstar_model_dir
pprime_model_dir = args.pprime_model_dir
pstar_data_dir = args.pstar_data_dir
pprime_data_dir = args.pprime_data_dir

# === accelerator init ===
accelerator = Accelerator()
device = accelerator.device
set_seed(seed)

# === load tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B")
tokenizer.pad_token = tokenizer.eos_token

# === data loading ===
def load_responses_from_dir(directory, max_samples):
    responses = []
    files = sorted(glob.glob(os.path.join(directory, "*.json")))
    for file in files:
        if accelerator.is_main_process:
            print("file:", file)
        with open(file, "r") as f:
            try:
                data_list = json.load(f)
                for item in data_list:
                    if "response" in item:
                        responses.append(item["response"])
                    if len(responses) >= max_samples:
                        return responses
            except json.JSONDecodeError:
                continue
    return responses

@torch.no_grad()
def compute_log_likelihood(model, texts, batch_size=8, tag=""):
    model.eval()
    total_log_likelihood = 0.0
    total_tokens = 0

    log_table = wandb.Table(columns=["text", "log_likelihood", "token_count"])

    dataloader = DataLoader(texts, batch_size=batch_size)
    for batch in dataloader:
        # Tokenize with padding/truncation
        encodings = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                              max_length=max_length).to(device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Compute per-token loss
        outputs = model(**encodings, labels=input_ids)
        logits = outputs.logits

        # Shift logits and labels for causal LM
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()

        # Compute log-probs and select the label probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        shift_log_probs = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        # Masked sum of log-probs
        per_sample_log_likelihood = (shift_log_probs * shift_mask).sum(dim=1)  # [batch]
        token_counts = shift_mask.sum(dim=1)  # [batch]

        # Aggregate totals
        total_log_likelihood += per_sample_log_likelihood.sum().item()
        total_tokens += token_counts.sum().item()

        for text, ll, n_tok in zip(batch, per_sample_log_likelihood.tolist(), token_counts.tolist()):
            log_table.add_data(text, ll, n_tok)

    mean_log_likelihood = total_log_likelihood / total_tokens if total_tokens > 0 else float("-inf")
    perplexity = exp(-mean_log_likelihood)

    return mean_log_likelihood, perplexity, log_table


# === wandb setup ===
if accelerator.is_main_process:
    wandb.login(key="WANDB_KEY")
    wandb.init(
        project="smollm2-pretraining",
        name=f"eval-p*-{os.path.basename(pstar_model_dir)}-pprime-{os.path.dirname(pprime_model_dir)}-seed{seed}",
        resume="allow"
    )

# === Recall  ===
if accelerator.is_main_process:
    print("Loading samples from p* (A)...")
    samples_from_p_star = load_responses_from_dir(pstar_data_dir, sample_size)
    
    print ('len(samples_from_p_star) : ', len(samples_from_p_star))

    print("Loading model p' for Recall...")
    model_p_prime = AutoModelForCausalLM.from_pretrained(pprime_model_dir).to(device)

    print("Computing Recall...")
    recall_ll, recall_ppl, recall_table = compute_log_likelihood(model_p_prime, samples_from_p_star, batch_size, tag="recall")
    
    print ('recall_ll : ', recall_ll)

    del model_p_prime
    torch.cuda.empty_cache()

# === Precision ===
if accelerator.is_main_process:
    print("Loading samples from p' (B)...")
    samples_from_p_prime = load_responses_from_dir(pprime_data_dir, sample_size)

    print ('len(samples_from_p_prime) : ', len(samples_from_p_prime))

    print("Loading model p* for Precision...")
    model_p_star = AutoModelForCausalLM.from_pretrained(pstar_model_dir).to(device)

    print("Computing Precision...")
    precision_ll, precision_ppl, precision_table = compute_log_likelihood(model_p_star, samples_from_p_prime, batch_size, tag="precision")
    
    print ('precision_ll : ', precision_ll)

    del model_p_star
    torch.cuda.empty_cache()

# === wandb logging and print ===
if accelerator.is_main_process:
    wandb.log({
        "precision_log_likelihood": precision_ll,
        "precision_perplexity": precision_ppl,
        "recall_log_likelihood": recall_ll,
        "recall_perplexity": recall_ppl,
        "sample_size": sample_size,
        "batch_size": batch_size,
        "max_length": max_length,
    })

    wandb.log({"precision_table": precision_table})
    wandb.log({"recall_table": recall_table})

    print(f"[Precision]  log-likelihood: {precision_ll:.4f} | Perplexity: {precision_ppl:.4f}")
    print(f"[Recall]     log-likelihood: {recall_ll:.4f} | Perplexity: {recall_ppl:.4f}")

    wandb.finish()

accelerator.wait_for_everyone()
