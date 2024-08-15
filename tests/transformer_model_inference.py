import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline

import cProfile
import pstats
import io

def transformer_generate(model_name, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_tokens = tokenizer.encode(prompt, return_tensors="pt")

    print(input_tokens)

    output_tokens = model.generate(
        input_tokens,
        max_length=7,
        do_sample=False,
        temperature=1.0,
        top_p=1.0
    )

    print(output_tokens[0].tolist())
    return output_tokens[0].tolist()

def print_cProfile(model_name, prompt):
    pr = cProfile.Profile()
    pr.enable()
    transformer_generate(model_name, prompt)
    pr.disable()

    # Print the profiling results
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())


model_name = "meta-llama/Llama-2-7b-hf"
prompt = "My favorite music is "


transformer_generate(model_name, prompt)
print_cProfile(model_name, prompt)
