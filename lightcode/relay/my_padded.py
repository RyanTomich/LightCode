# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "meta-llama/Llama-2-7b-hf"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)

# model.eval()
# tokenizer.pad_token = tokenizer.eos_token

# device = torch.device("cpu")
# model = model.to(device)

prompt = "In a galaxy far, far away "

# ###### Prefill Padded
# inputs  = tokenizer(prompt, padding='max_length', truncation=True, max_length=15, return_tensors='pt')

# input_ids = inputs["input_ids"].to(device)
# attention_mask = inputs["attention_mask"].to(device)

# with torch.no_grad():
#     outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

# logits = outputs[0]
# past_key_values = outputs[1]

# last_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

# print(last_token_id)
# print(past_key_values[0][0].shape)

# ###### prefill unpadded
# inputs  = tokenizer(prompt, return_tensors='pt')

# input_ids = inputs["input_ids"].to(device)
# attention_mask = inputs["attention_mask"].to(device)

# with torch.no_grad():
#     outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)

# logits = outputs[0]
# past_key_values = outputs[1]

# last_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

# print(last_token_id)
# print(past_key_values[0][0].shape)


import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Load model and tokenizer
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7B-HF")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7B-HF")

tokenizer.pad_token = tokenizer.eos_token

# Encode sequences with padding
non_padded_ids = tokenizer.encode(prompt, return_tensors="pt", padding=False)
padded_ids = tokenizer.encode(prompt, return_tensors="pt", padding="max_length", max_length=512)

print(non_padded_ids)
print(padded_ids)

# Generate outputs
non_padded_output = model.generate(non_padded_ids, max_new_tokens=10)
padded_output = model.generate(padded_ids, max_new_tokens=10)

print(non_padded_output[0])
print(padded_output[0])

# Decode outputs
non_padded_output_text = tokenizer.decode(non_padded_output[0], skip_special_tokens=True)
padded_output_text = tokenizer.decode(padded_output[0], skip_special_tokens=True)

# Print outputs
print("Non-padded output:", non_padded_output_text)
print("Padded output:", padded_output_text)
