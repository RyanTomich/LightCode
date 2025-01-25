"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

scripted_model = torch.jit.trace(
    model,
    example_inputs=(torch.zeros(1, 1, dtype=torch.int64),),
    strict=False
)

compiled_model = torch.compile(scripted_model)

# Input text
input_text = "What is the meaning of life?"
inputs = tokenizer(input_text, return_tensors="pt").to("cpu")

# Generate text using the optimized model
outputs = compiled_model.generate(
    inputs["input_ids"],
    max_new_tokens=50,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
"""

import torch
from transformers import LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.fx import symbolic_trace


class CustomMatMul:
    def __init__(self, size_threshold):
        self.size_threshold = size_threshold

    def matmul(self, a, b):
        num_elements = a.numel() + b.numel()

        if torch.cuda.is_available() and num_elements > self.size_threshold:
            a = a.to("cuda")
            b = b.to("cuda")
            result = torch.matmul(a, b)
            return result

        return torch.matmul(a.cpu(), b.cpu())


model_name = "meta-llama/Llama-2-7b-hf"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

size_threshold = 10000
custom_backend = CustomMatMul(size_threshold=size_threshold)

input_text = "What is your favorite music?"
inputs = tokenizer(input_text, return_tensors="pt").to("cpu")
