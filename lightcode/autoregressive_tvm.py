import tvm

from tvm import relay
from tvm.relay import op
from tvm.relay import frontend

from tvm.contrib import graph_runtime
from tvm.contrib import graph_executor
from tvm.contrib import utils

import onnx
import numpy as np
import os



import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
def transformer_test(model_name, prompt):

    model = AutoModelForCausalLM.from_pretrained(model_name, torchscript=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model = model.eval()  # Change to eval mode

    gen_tokens = model.generate(input_ids, do_sample=False, max_new_tokens=1)
    return(gen_tokens)
    gen_text = tokenizer.batch_decode(gen_tokens)[0]
    # print(gen_text)


def transformer_torch_to_onnx(model_name):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model_name_save = model_name.split("/", 1)[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torchscript=True)

    dummy_input_ids = torch.randint(0, 1000, (1, 5)) # arbatrary, shapes are dynamic
    dummy_past_key_values = [torch.zeros(1, 1, 1, 1) for _ in range(2)] # Adjust according to the model's KV cache requirements

    model_onnx = model.eval()

    onnx_file_path = f"models/{model_name_save}.onnx"

    if os.path.exists(onnx_file_path):
        print("already a model")
        return

    print("making new model")
    input_names = ["input_ids"]
    output_names = ["logits"]

    torch.onnx.export(
        model_onnx,
        (dummy_input_ids,),
        onnx_file_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"}
        },
        opset_version=16,
    )

def autoregressive_TVM(model_name, sequence_len):
    '''must have onnx model and inputs'''
    model_name_save = model_name.split("/", 1)[-1]
    path_lib = f"models/{model_name_save}_{sequence_len}.tar"
    print(path_lib)

    onnx_model_path = f"models/{model_name_save}.onnx"
    onnx_model = onnx.load(onnx_model_path)

    input_names = [input.name for input in onnx_model.graph.input]
    input_shapes = {input.name: [dim.dim_value for dim in input.type.tensor_type.shape.dim] for input in onnx_model.graph.input}

    print("Inputs:")
    for name in input_names:
        print(f"Name: {name}, Shape: {input_shapes[name]}")

    if os.path.exists(path_lib):
        print("already a .tar")
        return

    onnx_model_path = f"models/{model_name_save}.onnx"
    onnx_model = onnx.load(onnx_model_path)

    shape_dict = {
        "input_ids": (1, sequence_len),
    }

    mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

    config = {"relay.FuseOps.max_depth": 0}

    target = "llvm"
    with tvm.transform.PassContext(opt_level=0, config=config):
        lib = relay.build(mod, target=target, params=params)

    lib.export_library(path_lib)

def autoregressive_TVM_inference(model_name, prompt):
    model_name_save = model_name.split("/", 1)[-1]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.numpy()  # Convert to numpy array
    print("Input IDs shape:", input_ids.shape)

    path_lib = f"models/{model_name_save}.tar"
    path_lib = f"models/{model_name_save}_{len(input_ids[0])}.tar"
    loaded_lib = tvm.runtime.load_module(path_lib)

    ctx = tvm.cpu()
    module = graph_executor.GraphModule(loaded_lib["default"](ctx))

    module.set_input("input_ids", tvm.nd.array(input_ids.astype("int64")))
    module.run()

    outputs = []
    num_outputs = len(module.get_output(0).shape)  # Assumes the output is a tensor; modify as needed
    for i in range(num_outputs):
        output = module.get_output(i).asnumpy()  # Retrieve output and convert to numpy array
        outputs.append(output)

    output = module.get_output(0).asnumpy()

    next_tok = np.argmax(output[0][-1])
    gen_tokens = np.append(input_ids, next_tok)

    return gen_tokens



model_name = "gpt2"
prompt = "My favorite music is Mr. Blue Sky"

tokenizer = AutoTokenizer.from_pretrained(model_name)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

transformer_torch_to_onnx(model_name)
autoregressive_TVM(model_name, len(input_ids[0]))
tvm_tokens = autoregressive_TVM_inference(model_name, prompt)
transformer_tokens = transformer_test(model_name, prompt)

print(tvm_tokens)
print(transformer_tokens)
