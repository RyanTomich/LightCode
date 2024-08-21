import torch
import torch.fx as fx
import torch_tensorrt  # requires GPU...

from typing import List

from transformers import AutoModelForCausalLM, AutoTokenizer, Cache

#####  Toy example #####
# @torch.compile
# def opt_foo2(x, y):
#     a = torch.sin(x)
#     b = torch.cos(y)
#     return a + b

# ans = opt_foo2(torch.randn(10, 10), torch.randn(10, 10))

# def llama_compile(prompt, model, tokenizer):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     inputs = inputs.to(device)
#     outputs = model(**inputs)
#     return outputs

##### Custom Backedn and graps #####


##### Transformers #####
"""
For GPU onlyw
https://pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_compile_transformers_example.html#torch-compile-transformer

jit - serializable model and static computation
trace - Pytorch to TorchScript
script - more dynamic than TorchScript
fx Graph - Graph IR
AOT Autograd
@torch.compile() # eager - interperted like python
@torch.compile(mode='graph') # - dynamic graphs with graph optimizations
"""


def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward


model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).eval().to("cpu")

dummy_input = tokenizer.encode("Hello, world!", return_tensors="pt")

enabled_precisions = {torch.float}
debug = True
workspace_size = 20 << 30
min_block_size = 7  # Lower value allows more graph segmentation
torch_executed_ops = {}

compilation_kwargs = {
    "enabled_precisions": enabled_precisions,
    "debug": debug,
    "workspace_size": workspace_size,
    "min_block_size": min_block_size,
    "torch_executed_ops": torch_executed_ops,
}

# torch._dynamo.list_backends()`
# ['cudagraphs', 'inductor', 'onnxrt', 'openxla', 'tvm']

optimized_model = torch.compile(
    model,
    backend="torch_tensorrt",
    dynamic=False,
    options=compilation_kwargs,
)
optimized_model(*dummy_input)

torch._dynamo.reset()


# prompt = "My favorite music is "
# outputs = ???
# logits = outputs.logits
# kv_cache = outputs.past_key_values
# print(len(kv_cache ))
# print(kv_cache[0][0].shape)
