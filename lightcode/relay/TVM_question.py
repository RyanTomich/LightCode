import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import onnx
import tvm
from tvm import relay

# def get_onnx_io(onnx_model):
#     graph = onnx_model.graph

#     in_shape_names = []
#     for input_tensor in graph.input:
#         input_name = input_tensor.name
#         input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
#         in_shape_names.append(input_name)
#         print(f"in - {input_name}: {input_shape}")

#     for output_tensor in graph.output:
#         output_name = output_tensor.name
#         output_shape = [
#             dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim
#         ]
#         print(f"out - {output_name}: {output_shape}")

#     return in_shape_names


model_name = "gpt2"
prompt = "My favorite music is "

os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torchscript=True)

inputs = tokenizer(prompt, return_tensors="pt")
model = model.eval()

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



onnx_model = onnx.load(onnx_model_path)
onnx.checker.check_model(onnx_model_path)

# shape_dict = {"input_ids": inputs["input_ids"].shape}
shape_dict = {"input_ids": (1, tvm.te.var("sequence_length"))}  # Dynamic shape



mod, params = relay.frontend.from_onnx(
    onnx_model, shape_dict
)

config = {"relay.FuseOps.max_depth": 0,}

target = tvm.target.Target("llvm", host="llvm")
with tvm.transform.PassContext(opt_level=0, config=config):
    lib = relay.build(mod, target=target, params=params)



with open(f'{model_name}_graph.json', "w") as f:
    f.write(lib.get_graph_json())

lib.export_library(f"{model_name}_lib.tar")

param_dict = lib.get_params()
with open(f"{model_name}_params.params", "wb") as f:
    f.write(relay.save_param_dict(param_dict))
