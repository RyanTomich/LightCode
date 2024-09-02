'''
[FAIL] Killed on loading param step
env: tvm_conda
'''

import torch
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


import onnx
import onnxruntime as ort
import numpy as np

import tvm
from tvm import relay
from tvm.relay import op
from tvm.contrib import graph_runtime
from tvm.contrib import graph_executor


def get_kv_cache(model, sequence_len):
    if hasattr(model.config, "n_head"):  # GPT-2 case
        num_heads = model.config.n_head
        head_dim = model.config.n_embd // model.config.n_head
    elif hasattr(model.config, "num_attention_heads"):  # LLaMA case
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // model.config.num_attention_heads
    else:
        raise ValueError(
            "Model config does not have expected attributes for attention heads or hidden size."
        )

    return torch.zeros(1, num_heads, sequence_len, head_dim)


# Pytorch
def tokenize_input(prompt, tokenizer):
    return tokenizer(prompt, return_tensors="pt")


def prefill_step(prompt, tokenizer):
    inputs = tokenize_input(prompt, tokenizer)
    input_ids = inputs["input_ids"].to(device)

    # Get model outputs, including past_key_values (key-value cache)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    print(f"prefill kv cache {outputs[1][0][0].shape}")

    logits = outputs[0]
    past_key_values = outputs[1]

    return logits, past_key_values


def decoder_step(last_token_id, past_key_values):
    last_token_id = torch.tensor([[last_token_id]], device=device)

    with torch.no_grad():
        outputs = model(last_token_id, past_key_values=past_key_values, use_cache=True)

    logits = outputs[0]
    past_key_values = outputs[1]

    print(f"decoder kv cache {past_key_values[0][0].shape}")

    return logits, past_key_values


def generate(prompt, tokenizer, num_tokens=10):
    inputs = tokenize_input(prompt, tokenizer)
    input_ids = inputs["input_ids"].to(device)

    logits, past_key_values = prefill_step(prompt, tokenizer)

    last_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
    generated_sequence = [last_token_id]

    for i in range(num_tokens):
        logits, past_key_values = decoder_step(last_token_id, past_key_values)
        last_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated_sequence.append(last_token_id)

    print(f"{input_ids} + {generated_sequence}")
    generated_text = tokenizer.decode(generated_sequence)
    return generated_text, last_token_id, past_key_values


# Exporting to onnx
def onnx_export_prefill(model):
    onnx_path =  "../models/llama_prefill.onnx"
    if os.path.exists(onnx_path):
        print("already a {onnx_path}")
        return

    # 10 sequence length is arbatrary. will make dynamic anyway
    dummy_input_ids = torch.randint(
        0, model.config.vocab_size, (1, 10), dtype=torch.int64
    ).to(device)

    key_val_names = []
    # for layer in range(model.config.n_layer):
    for layer in range(model.config.num_hidden_layers):
        key_val_names.append(f"past_k_{layer}")
        key_val_names.append(f"past_v_{layer}")

    model.eval()

    torch.onnx.export(
        model,
        (dummy_input_ids,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"] + key_val_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            **{name: {0: "batch_size", 2: "sequence_length"} for name in key_val_names},
        },
        opset_version=16,
    )


def onnx_export_decoder(model):

    onnx_path = "../models/llama_decoder.onnx"
    if os.path.exists(onnx_path):
        print("already a {onnx_path}")
        return

    class LlamaWithKVCache(torch.nn.Module):
        def __init__(self, llama_model):
            super(LlamaWithKVCache, self).__init__()
            self.llama_model = llama_model

        def forward(self, input_ids, past_key_values):
            output = self.llama_model(
                input_ids=input_ids, past_key_values=past_key_values
            )
            return output[0], output[1]

    model_with_kv_cache = LlamaWithKVCache(model)
    model_with_kv_cache.eval()

    # 10 sequence length is arbatrary. will make dynamic anyway
    dummy_last_token_id = torch.tensor([[50256]], device=device)
    dummy_last_token_id = torch.tensor(
        [[tokenizer.eos_token_id]], device=device
    )  # use end-of-scentence token
    dummy_past_key_values = [
        (get_kv_cache(model, 10), get_kv_cache(model, 10))
        for _ in range(model.config.num_hidden_layers)
    ]

    past_key_val_names = []
    past_key_val_out_names = []
    # for layer in range(model.config.n_layer):
    for layer in range(model.config.num_hidden_layers):
        past_key_val_names.append(f"past_k_{layer}")
        past_key_val_names.append(f"past_v_{layer}")
        past_key_val_out_names.append(f"past_k_{layer}_out")
        past_key_val_out_names.append(f"past_v_{layer}_out")

    torch.onnx.export(
        model_with_kv_cache,
        (
            dummy_last_token_id,
            dummy_past_key_values,
        ),
        onnx_path,
        input_names=["input_ids"] + past_key_val_names,
        output_names=["logits"] + past_key_val_out_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            **{
                name: {0: "batch_size", 2: "sequence_length"}
                for name in past_key_val_names
            },
            "logits": {0: "batch_size", 1: "sequence_length"},
            **{
                name: {0: "batch_size", 2: "sequence_length"}
                for name in past_key_val_out_names
            },
        },
        opset_version=16,
    )


def get_profile_prefill_decoder():
    session_options = ort.SessionOptions()
    session_options.enable_profiling = True  # Enable profiling here

    session = ort.InferenceSession(
        "../models/llama_prefill.onnx", sess_options=session_options
    )

    input_name = session.get_inputs()[0].name  # Get the input name
    dummy_input = {input_name: np.random.randn(1, 10).astype(np.int64)}

    outputs = session.run(None, dummy_input)
    profile_file = session.end_profiling()

    session = ort.InferenceSession(
        "../models/llama_decoder.onnx", sess_options=session_options
    )

    dummy_input_ids = np.array([[50256]], dtype=np.int64)  # dummy last token
    input_names = [input.name for input in session.get_inputs()]

    dummy_input = {name: value for name, value in zip(input_names[1:], outputs[1:])}
    dummy_input["last_token_id"] = np.array([[50256]], dtype=np.int64)

    session.run(None, dummy_input)

    profile_file = session.end_profiling()


# Exporting to TVM relay
def get_onnx_io(onnx_model):
    graph = onnx_model.graph

    in_shape_names = []
    for input_tensor in graph.input:
        input_name = input_tensor.name
        input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        in_shape_names.append(input_name)
        print(f"in - {input_name}: {input_shape}")

    for output_tensor in graph.output:
        output_name = output_tensor.name
        output_shape = [
            dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim
        ]
        print(f"out - {output_name}: {output_shape}")

    return in_shape_names


def onnx_to_relay_prefill(input_shape, opt_level=0):
    onnx_model_path = "../models/llama_prefill.onnx"
    onnx_model = onnx.load(onnx_model_path)

    # get_onnx_io(onnx_model)

    shape_dict = {"input_ids": input_shape}

    onnx.checker.check_model(onnx_model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    config = {
        "relay.FuseOps.max_depth": 0,
    }

    target = tvm.target.Target("llvm", host="llvm")
    with tvm.transform.PassContext(opt_level=opt_level, config=config):
        lib = relay.build(mod, target=target, params=params)

    return lib


def run_relay_prefill(lib, inputs):
    target = tvm.cpu()
    module = graph_executor.GraphModule(lib["default"](target))

    input_ids = tvm.nd.array(inputs["input_ids"].numpy())
    module.set_input("input_ids", input_ids)

    module.run()

    outputs = []
    num_outputs = module.get_num_outputs()
    outputs = [module.get_output(i).numpy() for i in range(num_outputs)]

    logits = outputs[0]

    last_token_logits = logits[0, -1, :]
    next_token_id = np.argmax(last_token_logits)

    print(f"{input_ids} + {next_token_id}")
    return next_token_id, outputs[1:]


def onnx_to_relay_decoder(kv_cache_shape, opt_level=0):
    onnx_model_path = "../models/llama_decoder.onnx"
    onnx_model = onnx.load(onnx_model_path)

    # get_onnx_io(onnx_model)

    shape_dict = {}
    # for layer in range(model.config.n_layer):
    for layer in range(model.config.num_hidden_layers):
        shape_dict[f"past_k_{layer}"] = kv_cache_shape
        shape_dict[f"past_v_{layer}"] = kv_cache_shape

    shape_dict["input_ids"] = (1, 1)

    onnx.checker.check_model(onnx_model_path)
    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    config = {"relay.FuseOps.max_depth": 0}

    target = tvm.target.Target("llvm", host="llvm")
    with tvm.transform.PassContext(opt_level=opt_level, config=config):
        lib = relay.build(mod, target=target, params=params)

    return lib


def run_relay_decoder(lib, last_token_id, kv_cache):
    target = tvm.cpu()
    module = graph_executor.GraphModule(lib["default"](target))

    input_ids = tvm.nd.array(last_token_id)
    module.set_input("input_ids", input_ids)
    for layer in range(int(len(kv_cache) / 2)):
        past_k = tvm.nd.array(kv_cache[layer * 2])
        past_v = tvm.nd.array(kv_cache[layer * 2 + 1])
        module.set_input(f"past_k_{layer}", past_k)
        module.set_input(f"past_v_{layer}", past_v)

    module.run()

    # Get outputs
    outputs = []
    num_outputs = module.get_num_outputs()
    outputs = [module.get_output(i).numpy() for i in range(num_outputs)]

    logits = outputs[0]

    last_token_logits = logits[0, -1, :]
    next_token_id = np.argmax(last_token_logits)

    print(f"{input_ids} + {next_token_id}")
    return next_token_id, outputs[1:]


def save_relay(name, lib):
    graph_json_path = f'{name}_graph.json'
    with open(graph_json_path, "w") as f:
        f.write(lib.get_graph_json())


model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, torchscript=True)

model.eval()

tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cpu")
model = model.to(device)

prompt = "The future of AI is going to be"
inputs = tokenize_input(prompt, tokenizer)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

generated_text, last_token_id, past_key_values = generate(
    prompt, tokenizer, num_tokens=5
)

onnx_export_prefill(model)
# onnx_export_decoder(model)

# # get_profile_prefill_decoder()

sequence_len = len(input_ids[0])
input_ids_shape = (1, sequence_len)

print(sequence_len)
print(input_ids_shape)

# kv_cache_shape = get_kv_cache(model, sequence_len).shape

prefill_lib = onnx_to_relay_prefill(input_ids_shape)
# decoder_lib = onnx_to_relay_decoder(kv_cache_shape)

prompt = "My favorite music is "
inputs = tokenize_input(prompt, tokenizer)

generated_text, last_token_id, past_key_values = generate(
    prompt, tokenizer, num_tokens=5
)

next_token_id, kv_cache = run_relay_prefill(prefill_lib, inputs)
# next_token_id, kv_cache = run_relay_decoder(decoder_lib, next_token_id, kv_cache)

save_relay("llama_2_7b_prefill", prefill_lib)
