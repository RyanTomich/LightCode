import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

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

def onnx_export_prefill(model, inputs):

    def prefill_step_for_export(input_ids):
        outputs = model(input_ids, use_cache=True)
        return outputs.logits, outputs.past_key_values

    input_ids = inputs["input_ids"].to(device)

    torch.onnx.export(
        model,  # The model you want to export
        (input_ids,),  # Provide the input to the model
        "prefill_step.onnx",  # Output ONNX file
        input_names=["input_ids"],  # Input name
        output_names=["logits", "past_key_values"],  # Outputs
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},  # Make input dynamic
            "logits": {0: "batch_size", 1: "sequence_length"},
            "past_key_values": {0: "batch_size", 2: "sequence_length"}  # Make past_key_values dynamic
        },
        opset_version=16   # You might need to adjust this depending on your ONNX version
    )

def onnx_export_decoder(model, kv_sequence_lenght=1):

    def decoder_step_for_export(last_token_id, past_key_values):
        outputs = model(last_token_id, past_key_values=past_key_values, use_cache=True)
        return outputs.logits, outputs.past_key_values

    last_token_id = torch.tensor([[50256]], device=device)  # Example token

    # print(torch.zeros(1, model.config.n_head, kv_sequence_lenght, model.config.n_embd // model.config.n_head).to(device).shape)

    dummy_past_key_values = tuple(
        (torch.zeros(1, model.config.n_head, kv_sequence_lenght, model.config.n_embd // model.config.n_head).to(device),
        torch.zeros(1, model.config.n_head, kv_sequence_lenght, model.config.n_embd // model.config.n_head).to(device))
        for _ in range(model.config.n_layer)
    )

    torch.onnx.export(
        model,  # The model to export
        (last_token_id, dummy_past_key_values),  # Inputs to the model
        "decoder_step.onnx",  # Output ONNX file
        input_names=["last_token_id", "past_key_values"],  # Inputs
        output_names=["logits", "past_key_values"],  # Outputs
        dynamic_axes={
            "last_token_id": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "past_key_values": {0: "batch_size", 2: "sequence_length"},  # Make past_key_values dynamic
        },
        opset_version=16  # Adjust depending on your ONNX version
    )




# model_name = 'gpt2'

# tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# model = GPT2LMHeadModel.from_pretrained(model_name)
# model.eval()

# device = torch.device("cpu")
# model = model.to(device)

# prompt = "The future of AI is"
# inputs = tokenize_input(prompt)

# print(generate(prompt, num_tokens=20))

# onnx_export_prefill(model, inputs)
# onnx_export_decoder(model, kv_sequence_lenght = 10)


import onnx
import onnxruntime as ort
import numpy as np

session_options = ort.SessionOptions()
session_options.enable_profiling = True  # Enable profiling here

session = ort.InferenceSession("prefill_step.onnx", sess_options=session_options)

input_name = session.get_inputs()[0].name  # Get the input name
print(input_name)
dummy_input = {
    session.get_inputs()[0].name: np.random.randn(1, 10).astype(np.int64)
}

outputs = session.run(None, dummy_input)
profile_file = session.end_profiling()

session = ort.InferenceSession("decoder_step.onnx", sess_options=session_options)

dummy_input_ids = np.array([[50256]], dtype=np.int64)  # last_token_id
input_names = [input.name for input in session.get_inputs()]

dummy_input = {name: value for name, value in zip(input_names[1:], outputs[1:])}
dummy_input['last_token_id'] = np.array([[50256]], dtype=np.int64)

session.run(None, dummy_input)

profile_file = session.end_profiling()
