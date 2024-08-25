import torch
import time
import psutil


from transformers import LlamaForCausalLM, LlamaTokenizer
from torchao.quantization import quantize_, int8_weight_only

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

inference_time_after = measure_inference_time(model, input_ids)
inference_time_after = measure_inference_time(model, input_ids)

quantize_(model, int8_weight_only())

inference_time_after = measure_inference_time(model, input_ids)
