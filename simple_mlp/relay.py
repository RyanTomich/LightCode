import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import tvm
from tvm import relay
from tvm.contrib import graph_executor
import numpy as np

class LeNet300100(nn.Module):
    def __init__(self):
        super(LeNet300100, self).__init__()
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def onnx_to_relay(
    input_tensor, run=True, write=False, model_name="model", opt_level=0, config={}
):
    model_name_save = model_name.split("/", 1)[-1]
    model_onnx_path = f"{model_name_save}.onnx"
    model_onnx = onnx.load(model_onnx_path)

    shape_dict = {"input": tuple(input_tensor.shape)}
    onnx.checker.check_model(model_onnx)

    mod, params = relay.frontend.from_onnx(model_onnx, shape_dict)

    target = tvm.target.Target("llvm", host="llvm")
    with tvm.transform.PassContext(opt_level=opt_level, config=config):
        lib = relay.build(mod, target=target, params=params)

    if run:
        module = graph_executor.GraphModule(lib["default"](tvm.cpu()))
        tvm_input = tvm.nd.array(input_tensor.numpy())
        module.set_input("input", tvm_input)
        module.run()
        out = module.get_output(0).asnumpy()
        print("Output shape:", out.shape)

    if write:
        graph_json_path = f"{model_name_save}_graph.json"
        with open(graph_json_path, "w") as f:
            f.write(lib.get_graph_json())
        print(f"Wrote graph JSON to {graph_json_path}")

    return lib

if __name__ == "__main__":
    model = LeNet300100()
    model.eval()
    dummy_input = torch.randn(1, 1, 28, 28)

    torch.onnx.export(
        model,
        dummy_input,
        "lenet300100.onnx",
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=13
    )

    print("ONNX export complete. Running TVM conversion.")
    flat_input = dummy_input.view(1, 28 * 28)  # Match model’s internal flattening
    onnx_to_relay(flat_input, run=True, write=True, model_name="lenet300100")
