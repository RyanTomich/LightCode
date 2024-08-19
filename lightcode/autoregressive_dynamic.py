'''Failing'''

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import onnx
import onnxruntime as ort
import numpy as np

import tvm
from tvm import relay
from tvm.relay import op
from tvm.contrib import graph_runtime
from tvm.contrib import graph_executor

# Pytorch
def tokenize_input(prompt):
    return tokenizer(prompt, return_tensors="pt")

def prefill_step(prompt):
    inputs = tokenize_input(prompt)
    input_ids = inputs["input_ids"].to(device)

    # Get model outputs, including past_key_values (key-value cache)
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)

    logits = outputs.logits
    past_key_values = outputs.past_key_values  # Cache for decoder step

    return logits, past_key_values

def decoder_step(last_token_id, past_key_values):
    last_token_id = torch.tensor([[last_token_id]], device=device)

    with torch.no_grad():
        outputs = model(last_token_id, past_key_values=past_key_values, use_cache=True)

    logits = outputs.logits
    past_key_values = outputs.past_key_values  # Update cache

    return logits, past_key_values

def generate(prompt, num_tokens=10):
    inputs = tokenize_input(prompt)
    input_ids = inputs["input_ids"].to(device)

    logits, past_key_values = prefill_step(prompt)

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
    # def prefill_step_for_export(input_ids):
    #     outputs = model(input_ids, use_cache=True)
    #     return outputs.logits, outputs.past_key_values

    dummy_input_ids = torch.randint(0, model.config.vocab_size, (1, 10), dtype=torch.int64).to(device)


    key_val_names = []
    for layer in range(model.config.n_layer):
        key_val_names.append(f'past_k_{layer}')
        key_val_names.append(f'past_v_{layer}')

    model.eval()

    torch.onnx.export(
        model,
        (dummy_input_ids,),
        "prefill_step.onnx",
        input_names=["input_ids"],
        output_names=["logits"] + key_val_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            **{name: {0: "batch_size", 2: "sequence_length"} for name in key_val_names}
        },
        opset_version=16
    )

def onnx_export_decoder(model, dummy_last_token_id, dummy_past_key_values):
    # def decoder_step_for_export(last_token_id, past_key_values):
    #     outputs = model(last_token_id, past_key_values=past_key_values, use_cache=True)
    #     return outputs.logits, outputs.past_key_values

    # dummy_last_token_id = torch.tensor([[50256]], device=device)  # Example token

    # past_key_val_names = []
    # dummy_past_key_values = []
    # for layer in range(model.config.n_layer):
    #     past_key_val_names.append(f'past_k_{layer}')
    #     past_key_val_names.append(f'past_v_{layer}')
    #     dummy_past_key_values.append(torch.zeros(1, model.config.n_head, kv_sequence_lenght, model.config.n_embd // model.config.n_head).to(device))
    #     dummy_past_key_values.append(torch.zeros(1, model.config.n_head, kv_sequence_lenght, model.config.n_embd // model.config.n_head).to(device))

    torch.onnx.export(
        model,  # The model to export
        (dummy_last_token_id,) + tuple(dummy_past_key_values),
        "decoder_step.onnx",
        input_names=["last_token_id", "dummy_past_key_values"],
        output_names=["logits", "dummy_past_key_values"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},  # Make input dynamic
            **{name: {0: "batch_size", 2: "sequence_length"} for name in past_key_val_names}  # Dynamic past key-values for each layer
        },
        opset_version=16  # Adjust depending on your ONNX version
    )


def get_profile_prefill_decoder():
    session_options = ort.SessionOptions()
    session_options.enable_profiling = True  # Enable profiling here

    session = ort.InferenceSession("prefill_step.onnx", sess_options=session_options)

    input_name = session.get_inputs()[0].name  # Get the input name
    dummy_input = {
        input_name: np.random.randn(1, 10).astype(np.int64)
    }
    print(dummy_input)

    outputs = session.run(None, dummy_input)
    profile_file = session.end_profiling()

    session = ort.InferenceSession("decoder_step.onnx", sess_options=session_options)

    dummy_input_ids = np.array([[50256]], dtype=np.int64)  # dummy last token
    input_names = [input.name for input in session.get_inputs()]

    dummy_input = {name: value for name, value in zip(input_names[1:], outputs[1:])}
    dummy_input['last_token_id'] = np.array([[50256]], dtype=np.int64)

    session.run(None, dummy_input)

    profile_file = session.end_profiling()


# Exporting to TVM relay
def get_onnx_io(onnx_model):
    graph = onnx_model.graph

    for input_tensor in graph.input:
        input_name = input_tensor.name
        input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f'in - {input_name}: {input_shape}')

    for output_tensor in graph.output:
        output_name = output_tensor.name
        output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f'out - {output_name}: {output_shape}')

def onnx_to_relay_prefill(input_shape, opt_level=0):
    onnx_model_path = 'prefill_step.onnx'
    onnx_model = onnx.load(onnx_model_path)

    get_onnx_io(onnx_model)

    shape_dict = {"input_ids": input_shape}

    onnx.checker.check_model(onnx_model_path)
    mod, params = relay.frontend.from_onnx(
        onnx_model, shape_dict
    )

    config = {"relay.FuseOps.max_depth": 0}

    target = tvm.target.Target("llvm", host="llvm")
    with tvm.transform.PassContext(opt_level=opt_level, config=config):
        lib = relay.build(mod, target=target, params=params)

    return lib

def run_relay_prefill(lib, inputs):
    target = tvm.cpu()
    module = graph_executor.GraphModule(lib["default"](target))

    input_ids = tvm.nd.array(inputs["input_ids"].numpy())
    module.set_input("input_ids", input_ids)

    # attention_mask = tvm.nd.array(inputs["attention_mask"].numpy())
    # module.set_input("attention_mask", attention_mask)

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

def onnx_to_relay_decoder(kv_cache_shape, opt_level=0):
    onnx_model_path = 'decoder_step.onnx'
    onnx_model = onnx.load(onnx_model_path)

    get_onnx_io(onnx_model)

    # past_key_val_names = []
    # for layer in range(model.config.n_layer):
    #     past_key_val_names.append(f'past_k_val_{layer}')
    #     past_key_val_names.append(f'past_v_val_{layer}')

    # shape_dict

    return
    onnx.checker.check_model(onnx_model_path)
    mod, params = relay.frontend.from_onnx(
        onnx_model, shape_dict
    )

    config = {"relay.FuseOps.max_depth": 0}

    target = tvm.target.Target("llvm", host="llvm")
    with tvm.transform.PassContext(opt_level=opt_level, config=config):
        lib = relay.build(mod, target=target, params=params)

def run_relay_decoder():
    target = tvm.cpu()
    module = graph_executor.GraphModule(lib["default"](target))

    # Set inputs
    module.set_input("last_token_id", prev_gen_tok)
    for index, name in enumerate(shape_dict.keys()):
        module.set_input("name", kv_cache[index])


    # attention_mask = tvm.nd.array(inputs["attention_mask"].numpy())
    # module.set_input("attention_mask", attention_mask)

    # Run inference
    module.run()

    # Get outputs
    outputs = []
    num_outputs = len(module.get_output(0).shape)  # Assumes the output is a tensor; modify as needed
    outputs = [module.get_output(i).asnumpy() for i in range(num_outputs)]
    for out in outputs:
        print(out.shape)

    logits = outputs[0]

    last_token_logits = logits[0, -1, :]
    next_token_id = np.argmax(last_token_logits)

    print(f"{next_token_id}")
    return next_token_id, outputs[0]


model_name = 'gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()

device = torch.device("cpu")
model = model.to(device)

prompt = "The future of AI is"
inputs = tokenize_input(prompt)

generated_text, last_token_id, past_key_values = generate(prompt, num_tokens=5)

onnx_export_prefill(model)
# onnx_export_decoder(model, last_token_id, past_key_values)

# get_profile_prefill_decoder()

lib = onnx_to_relay_prefill(inputs.input_ids.shape)
next_token_id, kv_cache = run_relay_prefill(lib, inputs)

# kv_cache_shape = None
# lib = onnx_to_relay_decoder(kv_cache_shape)
