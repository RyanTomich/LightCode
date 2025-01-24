
import torch
import os

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
def prefill_step(model, prompt, tokenizer, device):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    logits = outputs[0]
    past_key_values = outputs[1]

    return logits, past_key_values


def decoder_step(model, device, last_token_id, past_key_values):
    last_token_id = torch.tensor([[last_token_id]], device=device)

    with torch.no_grad():
        outputs = model(last_token_id, past_key_values=past_key_values, use_cache=True)

    logits = outputs[0]
    past_key_values = outputs[1]

    return logits, past_key_values


def generate(model, prompt, tokenizer, device, num_tokens = 10):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generated_sequence = input_ids.tolist()[0]

    # prefill
    logits, past_key_values = prefill_step(model, prompt, tokenizer, device)
    last_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
    generated_sequence.append(last_token_id)

    # decoder
    for _ in range(num_tokens):
        logits, past_key_values = decoder_step(model, device, last_token_id, past_key_values)
        last_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()
        generated_sequence.append(last_token_id)

    generated_text = tokenizer.decode(generated_sequence)
    return generated_text, last_token_id, past_key_values


# Exporting to onnx
def onnx_export_prefill(model, device, save_name):
    onnx_path =  f"models/{save_name}_prefill.onnx"
    if os.path.exists(onnx_path): # save time if already exists
        print("already a {onnx_path}")
        return

    dummy_input_ids = torch.randint(
        0, model.config.vocab_size, (1, 10), dtype=torch.int64 # 10 sequence len is arbatrary
    ).to(device)

    key_val_names = []
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


def onnx_export_llama_decoder(model, device, save_name):
    onnx_path =  f"models/{save_name}_prefill.onnx"
    if os.path.exists(onnx_path): # save time if already exists
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

    dummy_last_token_id = torch.tensor([[50256]], device=device)
    dummy_last_token_id = torch.tensor(
        [[tokenizer.eos_token_id]], device=device
    )  # use end-of-scentence token
    dummy_past_key_values = [
        (get_kv_cache(model, 10), get_kv_cache(model, 10)) # 10 sequence length is arbatrary.
        for _ in range(model.config.num_hidden_layers)
    ]

    past_key_val_names = []
    past_key_val_out_names = []
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


def onnx_export_gpt2_decoder(model, device, save_name):
    onnx_path =  f"models/{save_name}_decoder.onnx"
    if os.path.exists(onnx_path): # save time if already exists
        print("already a {onnx_path}")
        return

    class GPT2WithKVCache(torch.nn.Module):
        def __init__(self, gpt2_model):
            super(GPT2WithKVCache, self).__init__()
            self.gpt2_model = gpt2_model

        def forward(self, input_ids, past_key_values):
            output = self.gpt2_model(
                input_ids=input_ids, past_key_values=past_key_values
            )
            return output.logits, output.past_key_values

    model_with_kv_cache = GPT2WithKVCache(model)
    model_with_kv_cache.eval()

    # 10 sequence length is arbatrary. will make dynamic anyway
    dummy_last_token_id = torch.tensor([[50256]], device=device)  # Example token
    # dummy_past_key_values = [(torch.zeros(1, model.config.n_head, 1, model.config.n_embd // model.config.n_head), torch.zeros(1, model.config.n_head, 1, model.config.n_embd // model.config.n_head)) for _ in range(model.config.n_layer)]
    dummy_past_key_values = [
        (get_kv_cache(model, 10), get_kv_cache(model, 10))
        for _ in range(model.config.n_layer)
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
        ),  # Pass input_ids and past_key_values as inputs
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


def onnx_to_relay_prefill(input_shape, save_name, opt_level=0):
    onnx_model_path =  f"models/{save_name}_prefill.onnx"

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



if __name__ == "__main__":
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
    generated_text, last_token_id, past_key_values = generate(model, prompt, tokenizer, device, num_tokens = 5)
    print(generated_text)


    # Prefill
    # onnx_export_prefill(model)
    # inputs = tokenizer(prompt, return_tensors="pt")
    # input_ids = inputs["input_ids"]
    # attention_mask = inputs["attention_mask"]

    # sequence_len = len(input_ids[0])
    # input_ids_shape = (1, sequence_len)
    # kv_cache_shape = get_kv_cache(model, sequence_len).shape

    # Decoder
    # onnx_export_decoder(model)
    # inputs = tokenizer(prompt, return_tensors="pt")
    # input_ids = inputs["input_ids"]

    # sequence_len = len(input_ids[0])
    # input_ids_shape = (1, sequence_len)
    # kv_cache_shape = get_kv_cache(model, sequence_len).shape

    # decoder_lib = onnx_to_relay_decoder(kv_cache_shape)


    # prompt = "My favorite music is "
    # inputs = tokenizer(prompt, return_tensors="pt")


    # generated_text, last_token_id, past_key_values = generate(
    #     prompt, tokenizer, num_tokens=5
    # )

    # # next_token_id, kv_cache = run_relay_prefill(prefill_lib, inputs)
    # # next_token_id, kv_cache = run_relay_decoder(decoder_lib, next_token_id, kv_cache)

    # # save_relay("llama_2_7b_prefill", prefill_lib)
    # save_relay("llama_2_7b_decoder", decoder_lib)
