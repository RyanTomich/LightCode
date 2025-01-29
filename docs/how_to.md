---
layout: default
title: "How To"
nav_order: 4
---
# Index
- [Generating Relay Conputational Graph](#generating-relay-conputational-graph)
- [Optimization Frontend](#optimization-frontend)
- [Optimization Pipeline](#optimization-pipeline)
- [Adding a new model](#adding-a-new-model)
- [Adding a new hardware](#adding-a-new-hardware)


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

## The code, all together
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

# Optimization Frontend

## File Depndancys
```python
from lightcode import main
from lightcode import hardware
from lightcode import models
```

## Create and Register Hardware
First we must tell the simulator what type of hardware will be available. Our simplified model of hardware considers only a static, average clock cycle and core count. This is an example of registering a CPU, PHU, and GPU. More complex architectures would need aditional backend support.

```python
local_hardware = []
hardware.Hardware._hardware_reset()

CPU_AVERAGE_CLOCK = 3.208 * 10**9  # 60**9, 6
CPU_CORES = 1
local_hardware.append(hardware.CPU(CPU_AVERAGE_CLOCK, CPU_CORES))

PHU_MIN_CLOCK = 9.7 * 10**9  # 100**9, 10 Ghz
PHU_CORES = 1
PHU_MULTIPLEX = 20
local_hardware.append(hardware.PHU(PHU_MIN_CLOCK, PHU_CORES, PHU_MULTIPLEX))

GPC = 8  # Graphical Processing Clusters
TPC_per_GPC = 9  # Texture Processing Clusters/Graphical Processing Cluster
SM_per_TPC = 2  # Streaming multiprocessors / Texture Processing Cluster
fp32_CUDA_cores_per_SM = 128  # fp32_CUDA_cores / Streaming multiprocessor
TC_per_SM = 4  # Tensor Cores / Streaming multiprocessor
local_hardware.append(
    hardware.GPU(
        GPU_FP32_CLOCK, GPC, TPC_per_GPC, SM_per_TPC, fp32_CUDA_cores_per_SM, TC_per_SM
    )
)

available_hardware = hardware.initilize_hardware(local_hardware)
```

## Run Graph Search and Thresholds
Next we tell the optimizer what its optimizing for.
- `time`: total time of the computation
- `energy`: total energy consumption of the computation
- 'always_phu': debug tool that always selects photonics if available

graph search will run the optmizations and return stats about time, energy, and how many of the posiable nodes selected photonics for the given model and optmization.

Threshold evaluates all multi-node stacks and establishes at which sequence length the optimizer switches hardware.

```python
# select an optimization
# optimization = "time"
optimization = "energy"
# optimization = "always_phu"

results = main.graph_search(
    models.gpt2_prefill,
    optimization,
    available_hardware,
    moc_sequence_length=1400, # arbatrary number
    profiles=True,
    colect_data=True,
)

thresholds = main.threshold_search(
    models.gpt2_prefill,
    optimization,
    available_hardware,
)
```

## The code, all together
```python
from lightcode import main
from lightcode import hardware
from lightcode import models

CPU_AVERAGE_CLOCK = 3.208 * 10**9  # 60**9, 6
PHU_MIN_CLOCK = 9.7 * 10**9  # 100**9, 10 Ghz
GPU_FP32_CLOCK = 1.98 * 10**9  # 1.98 GHz
CPU_CORES = 1
PHU_CORES = 1
PHU_MULTIPLEX = 20

local_hardware = []
hardware.Hardware._hardware_reset()
local_hardware.append(hardware.CPU(CPU_AVERAGE_CLOCK, CPU_CORES))
local_hardware.append(hardware.PHU(PHU_MIN_CLOCK, PHU_CORES, PHU_MULTIPLEX))

GPC = 8  # Graphical Processing Clusters
TPC_per_GPC = 9  # Texture Processing Clusters/Graphical Processing Cluster
SM_per_TPC = 2  # Streaming multiprocessors / Texture Processing Cluster
fp32_CUDA_cores_per_SM = 128  # fp32_CUDA_cores / Streaming multiprocessor
TC_per_SM = 4  # Tensor Cores / Streaming multiprocessor
local_hardware.append(
    hardware.GPU(
        GPU_FP32_CLOCK, GPC, TPC_per_GPC, SM_per_TPC, fp32_CUDA_cores_per_SM, TC_per_SM
    )
)

available_hardware = hardware.initilize_hardware(local_hardware)

# optimization = "time"
optimization = "energy"
# optimization = "always_phu"

results = main.graph_search(
    models.gpt2_prefill,
    optimization,
    available_hardware,
    moc_sequence_length=1400, # arbatrary number
    profiles=True,
    colect_data=True,
)

thresholds = main.threshold_search(
    models.gpt2_prefill,
    optimization,
    available_hardware,
)
```

# Optimization Pipeline
Lets expose one layer of abstraction to give more control over individual nodes in the graph. This can be useful for further analysis.

# Adding a new model

# Adding a new hardware


[^1]: For each token generated by an autoregressive LLM, the output is a stochastic vector. This vector is the same length as the model's vocabulary and it represents the probability that each token is the correct next one. There are strategies like [top-k](https://www.ibm.com/docs/en/watsonx/saas?topic=lab-model-parameters-prompting#:~:text=0.05-,Top%20K%20example,-Top%20K%20specifies) and [top-p](https://www.ibm.com/docs/en/watsonx/saas?topic=lab-model-parameters-prompting#:~:text=Top%20K%20%3D%201.-,Top%20P%20example,-Top%20P%20specifies) that decides which of these will be the next token. There are other parameters like [Temperature](https://www.ibm.com/docs/en/watsonx/saas?topic=lab-model-parameters-prompting#:~:text=reset%20to%200.-,Temperature%20example,-The%20temperature%20setting) that affect this decision as well. Greedy simply selects the index of the maximum(i.e. the highest probability next token). This makes LLm deterministic which is useful for testing. It is not good for getting interesting responses.
