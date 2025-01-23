#include <torch/extension.h>

torch::Tensor custom_matmul(torch::Tensor a, torch::Tensor b){
    a = a.contiguous();
    b = b.contiguous();

    return torch::matmul(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_matmul", &custom_matmul, "Custom Matmul (CPU)");
}
