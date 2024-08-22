# Kernel Launching and Dynamic Kernel Selection for LLM’s with Relay

Hi! I am working on using TVM to compile LLMs. Here, the prompt is specified in the compilation stage. The PyTorch and ONNX models allow for dynamic shapes, but the resulting TVM graph appears to be for the static shape of the prompt. Here is my workflow:

Imports
```
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import onnx
import tvm
from tvm import relay
```

Take model from Transformer Library
```
model_name = "gpt2"
prompt = "My favorite music is "

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torchscript=True)

inputs = tokenizer(prompt, return_tensors="pt")
model = model.eval()
```

Export to onnx
```
onnx_model_path = f"{model_name}.onnx"

key_val_names = []
for layer in range(model.config.num_hidden_layers):
    key_val_names.append(f"past_k_{layer}")
    key_val_names.append(f"past_v_{layer}")

torch.onnx.export(
    model,
    (inputs["input_ids"],),
    onnx_model_path,
    input_names=["input_ids"],
    output_names=["logits"] + key_val_names,
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        **{name: {0: "batch_size", 2: "sequence_length"} for name in key_val_names},
    },
    opset_version=16,
)
```

Build to TVM
```
onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model_path)

shape_dict = {"input_ids": inputs["input_ids"].shape}

mod, params = relay.frontend.from_onnx(
    onnx_model, shape_dict
)

config = {"relay.FuseOps.max_depth": 0,}

target = tvm.target.Target("llvm", host="llvm")
with tvm.transform.PassContext(opt_level=0, config=config):
    lib = relay.build(mod, target=target, params=params)
```
        * Note, opt_level=0 and relay.FuseOps.max_depth: 0 because i want access to the underlying dense and matmul operations without packing or operator fusion. (please comment if there is a better way)

Export
```
with open(f'{model_name}_graph.json', "w") as f:
    f.write(lib.get_graph_json())

lib.export_library(f"{model_name}_lib.tar")

param_dict = lib.get_params()
with open(f"{model_name}_params.params", "wb") as f:
    f.write(relay.save_param_dict(param_dict))
```


I was wondering some things:

1) Once I have the Relay graph, the library.tar, and params, I want to replace some of the operations (like dense) with my own and do custom scheduling. Is it possible to take these nodes and **run/kernel launch each node individually**? How can I take control of the kernel launch?

2) LLMs have unknown input lengths during prefill and changing kv-cache during the decoder phase. Is there a way for TVM to handle the dynamic shapes? Alternatively, does TVM support compiling multiple kernels and doing **dynamic kernel selection** at runtime depending on input size.

3) Trying to do everything in AOT-compilation, does TVM support **creating kernels for a range or sizes** (during dense or matmul, for example)?

        * Note, I have heard about TVM Unity and Relax that are currently under development, but I have not seen how to use those tools in this way.

Thanks for any guidance on how to solve these problems or improve the workflow.
