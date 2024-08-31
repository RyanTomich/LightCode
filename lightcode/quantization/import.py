import torch
import time
import psutil

import onnx
import tvm
from tvm import relay
from tvm.relay import op
from tvm.contrib import graph_runtime
from tvm.contrib import graph_executor

from transformers import LlamaForCausalLM, LlamaTokenizer
# from torchao.quantization import quantize_, int8_weight_only

def measure_inference_time(model, input_ids, num_runs=10):
    # Warm-up to ensure fair timing
    with torch.no_grad():
        _ = model(input_ids)

    # Measure inference time
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_ids)
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    print(avg_time)
    return avg_time


model_name = "meta-llama/Llama-2-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, torchscript=True)

model.eval()

input_text = "Once upon a time"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# inference_time_after = measure_inference_time(model, input_ids)

# quantize_(model, int8_weight_only())

# inference_time_after = measure_inference_time(model, input_ids)

# prompt = "My favorite music is "
# inputs = tokenizer(prompt, return_tensors="pt")

# torch.onnx.export(
#     model,
#     (inputs["input_ids"], inputs["attention_mask"]),
#     "quantized_model.onnx",
#     input_names=["input_ids", "attention_mask"],
#     output_names=["logits"],
#     dynamic_axes={
#         "input_ids": {0: "batch_size", 1: "sequence"},
#         "attention_mask": {0: "batch_size", 1: "sequence"},
#         "logits": {0: "batch_size", 1: "sequence"},
#     },
# )

model_onnx_path = f"quantized_model.onnx"
model_onnx = onnx.load(model_onnx_path)

shape_dict = {"input_ids": input_ids.shape, "attention_mask": input_ids.shape}

onnx.checker.check_model(model_onnx_path)
mod, _ = relay.frontend.from_onnx(
    model_onnx, shape_dict
)  # <class 'tvm.ir.module.IRModule'>

# Export model graph parts
config = {"relay.FuseOps.max_depth": 0,}
target = tvm.target.Target("llvm", host="llvm")
with tvm.transform.PassContext(opt_level=0, config=config):
    lib = relay.build(mod, target=target)

graph_json_path = f"quatnized_model.json"
with open(graph_json_path, "w") as f:
    f.write(lib.get_graph_json())

'''
Killed when doing mod, params
'''
