import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import random
import uuid
from huggingface_hub import login

# huggingface login
login('HUGGINGFACE_ID')

# model and tokenizer
model_name = "HuggingFaceTB/SmolLM2-1.7B"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print (tokenizer.bos_token, tokenizer.pad_token, tokenizer.eos_token)

# generation setup
num_samples = 1_000_000_000  # num of samples
temp = 1.0  
top_p = 1.0  
top_k = model.config.vocab_size  
num_return_sequences = 1  
batch_size = 256  

# input prompt
inputs = tokenizer(["The"] * batch_size, return_tensors="pt").to(device)

generated_data = []
for i in range(0, num_samples, batch_size):
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=True, 
            max_length = 512,
            temperature=temp,
            top_p=top_p, 
            top_k=top_k,
            num_return_sequences=num_return_sequences,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    
    for j in range(batch_size):
        responses = output[j * num_return_sequences : (j + 1) * num_return_sequences]
        decoded_responses = [tokenizer.decode(res, skip_special_tokens=True) for res in responses]
        for response in decoded_responses:
            # print ('response : ', response)
            generated_data.append({"prompt": "The", "response": response})

    # save data (every 100,000)
    if len(generated_data) >= 100_000:
        random_id = uuid.uuid4().hex  # generate random id
        output_file = f"generated_data/generated_data_part_{random_id}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(generated_data, f, ensure_ascii=False, indent=4)
        
        generated_data = [] 

    print ('len(generated_data) : ', len(generated_data))

# last data generation
if generated_data:
    random_id = uuid.uuid4().hex 
    output_file = f"generated_data/generated_data_part_{random_id}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(generated_data, f, ensure_ascii=False, indent=4)

print("Data generation complete. Files saved in chunks for efficient storage.")
