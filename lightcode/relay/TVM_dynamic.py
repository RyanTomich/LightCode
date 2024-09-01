import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import onnx
import tvm
from tvm import relay
from tvm.contrib import graph_executor

import autoregressive_gpt2 as ar_gpt2

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

device = torch.device("cpu")
model = model.to(device)

prompt = "My favorite music is "
inputs = ar_gpt2.tokenize_input(prompt, tokenizer)

dummy_input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

torch.onnx.export(
    model, dummy_input_ids, "gpt2.onnx",
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {1: 'sequence_length'}}
)

onnx_model = onnx.load("gpt2.onnx")

for input_tensor in onnx_model.graph.input:
    shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    print(f"Name: {input_tensor.name}, Shape: {shape},")


mod, params = relay.frontend.from_onnx(onnx_model)
# shape_dict = {"input_ids": (1, relay.Any())}
# mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

config = {
    "relay.FuseOps.max_depth": 0,
    "relay.backend.use_auto_scheduler": True,
}

# target = tvm.target.Target("llvm", host="llvm")
target = "llvm"
# disabled_pass=["AlterOpLayout"]
with tvm.transform.PassContext(opt_level=0, config=config):
    lib = relay.build(mod, target=target, params=params)


dev = tvm.cpu(0)
module = graph_executor.GraphModule(lib["default"](dev))

import numpy as np
input_data = np.random.randn(1, 20).astype("int64")  # Example input with sequence length of 20

module.set_input("input", input_data)
module.run()

output = module.get_output(0).asnumpy()
print(output)

'''
  Check failed: (pval != nullptr) is false: Cannot allocate memory symbolic tensor shape [1, ?]
'''
