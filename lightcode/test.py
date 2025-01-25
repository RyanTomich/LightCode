import torch
from lightcode import relay


# import the model from huggingface to TorchScript
from transformers import LlamaForCausalLM, LlamaTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, torchscript=True)

# Correct the model settings
model.eval()
tokenizer.pad_token = tokenizer.eos_token

# set to device
device = torch.device("cpu")
model = model.to(device)

# Test generation functionality
prompt = "The future of AI is going to be"
generated_text, last_token_id, past_key_values = relay.generate(
    model, prompt, tokenizer, device, num_tokens=5
)
print(generated_text)


# Prefill
save_name = model_name.split("/", 1)[-1]
relay.onnx_export_prefill(model, device, save_name)

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
sequence_len = len(input_ids[0])
input_ids_shape = (1, sequence_len)

prefill_lib = relay.onnx_to_relay_prefill(input_ids_shape, save_name)
relay.save_relay(f"{save_name}_prefill", prefill_lib)

# Decoder
save_name = model_name.split("/", 1)[-1]
relay.onnx_export_llama_decoder(model, device, save_name)

kv_cache_shape = relay.get_kv_cache(model, sequence_len).shape

decoder_lib = relay.onnx_to_relay_decoder(kv_cache_shape)
relay.save_relay(f"{save_name}_prefill", decoder_lib)


# Testing
generated_text, last_token_id, past_key_values = relay.generate(
    prompt, tokenizer, num_tokens=5
)

next_token_id, kv_cache = relay.run_relay_prefill(prefill_lib, inputs)
next_token_id, kv_cache = relay.run_relay_decoder(decoder_lib, next_token_id, kv_cache)
