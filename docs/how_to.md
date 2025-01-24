---
layout: default
title: "How To"
nav_order: 4
---

# Generating Relay Conputational Graph

## setup
Switch your conda enviroment with
```bash
conda activate tvm_conda
```

## Importing a model to pytorch
Running inference on pytorch

```python
import torch
import relay as lc_relay

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
generated_text, last_token_id, past_key_values = lc_relay.generate(model, prompt, tokenizer, device, num_tokens = 5)
print(generated_text)
```

## Model to Relay

### Prefill Stage
```python
save_name = model_name.split("/", 1)[-1]
lc_relay.onnx_export_prefill(model, device, save_name)

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
sequence_len = len(input_ids[0])
input_ids_shape = (1, sequence_len)

prefill_lib = lc_relay.onnx_to_relay_prefill(input_ids_shape, save_name)
lc_relay.save_relay(f"{save_name}_prefill", prefill_lib)

# Testing. Should match pytorch execution next token
next_token_id, kv_cache = lc_relay.run_relay_prefill(prefill_lib, inputs)
```
- CAUTION: This will take a long time to run depending on your computer and internet connection.
- CAUTION: This will create lots of large files (300+ of 100+ mb). These are the weights of the model.

After the weight files are removed, you should have 2 files remaining.
- `{model})_prefill.onnx` : model definition that used to link to the weights
- `Opt0_{model}_prefill_graph.json` : prefill computational Graph


### Decoder Stage
This is significantly more difficult and may require more custom functions to extract depending on your model.
```python
save_name = model_name.split("/", 1)[-1]
lc_relay.onnx_export_llama_decoder(model, device, save_name)

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
sequence_len = len(input_ids[0])
kv_cache_shape = lc_relay.get_kv_cache(model, sequence_len).shape

decoder_lib = lc_relay.onnx_to_relay_decoder(kv_cache_shape)
lc_relay.save_relay(f"{save_name}_prefill", decoder_lib)
```
- The `onnx_export_llama_decoder(model, device, save_name)` is a custom function in the `relay.py` file. There is an example for Llama-2-7b-hf and gpt2.

After the weight files are removed, you should have 2 files remaining.
- `{model})_decoder.onnx` : model definition that used to link to the weights
- `Opt0_{model}_decoder_graph.json` : decoder computational Graph


### Testing and cleanup
If the prefill and decoder stages are done together, you can validate results are consistent by comparing the pytorch inference to the TVM Relay inference. We are running Greedy[^1], so they should be the same.

```python
generated_text, last_token_id, past_key_values = lc_relay.generate(
    prompt, tokenizer, num_tokens=5
)
next_token_id, kv_cache = lc_relay.run_relay_prefill(prefill_lib, inputs)
next_token_id, kv_cache = lc_relay.run_relay_decoder(decoder_lib, next_token_id, kv_cache)
```
- NOTE: The fiels starting with `model.layers.#...` and `ONNX__MatMul_####` are only useful for validating the model. After the Computational Graph has been created, they are no longer needed and can be deleted.

We used TVM Rela IR to extract the conputational graphs of the meta-llama/Llama-2-7b-hf model.

## The code, all together.
```python
import torch
import relay as lc_relay

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
generated_text, last_token_id, past_key_values = lc_relay.generate(model, prompt, tokenizer, device, num_tokens = 5)
print(generated_text)


# Prefill
save_name = model_name.split("/", 1)[-1]
lc_relay.onnx_export_prefill(model, device, save_name)

inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
sequence_len = len(input_ids[0])
input_ids_shape = (1, sequence_len)

prefill_lib = lc_relay.onnx_to_relay_prefill(input_ids_shape, save_name)
lc_relay.save_relay(f"{save_name}_prefill", prefill_lib)

# Decoder
save_name = model_name.split("/", 1)[-1]
lc_relay.onnx_export_llama_decoder(model, device, save_name)

kv_cache_shape = lc_relay.get_kv_cache(model, sequence_len).shape

decoder_lib = lc_relay.onnx_to_relay_decoder(kv_cache_shape)
lc_relay.save_relay(f"{save_name}_prefill", decoder_lib)


# Testing
generated_text, last_token_id, past_key_values = lc_relay.generate(
    prompt, tokenizer, num_tokens=5
)

next_token_id, kv_cache = lc_relay.run_relay_prefill(prefill_lib, inputs)
next_token_id, kv_cache = lc_relay.run_relay_decoder(decoder_lib, next_token_id, kv_cache)
```

[^1]: For each token generated by an autoregressive LLM, the output is a stochastic vector. This vector is the same length as the model's vocabulary and it represents the probability that each token is the correct next one. There are strategies like [top-k](https://www.ibm.com/docs/en/watsonx/saas?topic=lab-model-parameters-prompting#:~:text=0.05-,Top%20K%20example,-Top%20K%20specifies) and [top-p](https://www.ibm.com/docs/en/watsonx/saas?topic=lab-model-parameters-prompting#:~:text=Top%20K%20%3D%201.-,Top%20P%20example,-Top%20P%20specifies) that decides which of these will be the next token. There are other parameters like [Temperature](https://www.ibm.com/docs/en/watsonx/saas?topic=lab-model-parameters-prompting#:~:text=reset%20to%200.-,Temperature%20example,-The%20temperature%20setting) that affect this decision as well. Greedy simply selects the index of the maximum(i.e. the highest probability next token). This makes LLm deterministic which is useful for testing. It is not good for getting interesting responses.
