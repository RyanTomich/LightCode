import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline

def transformer_generate(model_name, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    input_tokens = tokenizer.encode(prompt, return_tensors="pt")

    print(input_tokens)

    output_tokens = model.generate(
        input_tokens,
        max_length=20,
        do_sample=False,
        temperature=1.0,
        top_p=1.0
    )

    print(output_tokens[0].tolist())
    return output_tokens[0].tolist()

model_name = "meta-llama/Llama-2-7b-hf"
prompt = "My favorite music is "

transformer_generate(model_name, prompt)
