import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

# Move model to CUDA if available
device = torch.device("cpu")
model = model.to(device)

# Function to tokenize input
def tokenize_input(prompt):
    return tokenizer(prompt, return_tensors="pt")

# Prefill step: forward pass with the input prompt
def prefill_step(prompt):
    inputs = tokenize_input(prompt)
    input_ids = inputs["input_ids"].to(device)

    # Get model outputs, including past_key_values (key-value cache)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    logits = outputs.logits
    past_key_values = outputs.past_key_values  # Cache for decoder step

    return logits, past_key_values

# Decoder step: generate next token using cache
def decoder_step(last_token_id, past_key_values):
    # Input only the last token
    last_token_id = torch.tensor([[last_token_id]], device=device)

    # Pass the last token and cache (past_key_values)
    with torch.no_grad():
        outputs = model(last_token_id, past_key_values=past_key_values, use_cache=True)

    logits = outputs.logits
    past_key_values = outputs.past_key_values  # Update cache

    return logits, past_key_values

# Function to generate tokens autoregressively
def generate(prompt, num_tokens=10):
    # Step 1: Prefill step with initial prompt
    logits, past_key_values = prefill_step(prompt)

    # Get the token of the last generated word
    last_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
    generated_sequence = [last_token_id]

    # Step 2: Decoder steps
    for _ in range(num_tokens - 1):
        logits, past_key_values = decoder_step(last_token_id, past_key_values)

        # Get the token of the next word and update the last token
        last_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated_sequence.append(last_token_id)

    # Decode the generated sequence
    generated_text = tokenizer.decode(generated_sequence)
    return generated_text

# Test inference with an arbitrary prompt
prompt = "The future of AI is"
generated_text = generate(prompt, num_tokens=10)
print(f"Generated text: {generated_text}")
