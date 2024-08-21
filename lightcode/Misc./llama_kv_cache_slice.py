import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Cache


def tokenize_input(prompt, tokenizer, pad=False):
    return tokenizer(prompt, return_tensors="pt")


def logits_to_token(logits):
    do_sample = global_sample
    if do_sample:
        probs = F.softmax(logits[:, -1, :], dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits[:, -1, :], dim=-1)


def inference_generate(model, tokenizer, prompt):
    inputs = tokenize_input(prompt, tokenizer)
    inputs_no_pad, pad_to = get_pad_to(prompt, tokenizer)

    output_tokens = model.generate(
        **inputs,
        max_length=pad_to + global_sliding,
        do_sample=global_sample,
        temperature=None,
        top_p=None,
    )
    print(f"{output_tokens[0]}")

    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print(generated_text)
    return generated_text


def inference_prefill_decoder(model, tokenizer, prompt):
    def prefill_step(prompt, tokenizer):
        inputs = tokenize_input(prompt, tokenizer)
        input_ids = inputs["input_ids"].to(device)

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)

        print(f"prefill kv cache {outputs[1][0][0].shape}")

        logits = outputs[0]
        past_key_values = outputs[1]

        return logits, past_key_values

    def decoder_step(last_token_id, past_key_values):
        last_token_id = torch.tensor([[last_token_id]], device=device)

        with torch.no_grad():
            outputs = model(
                last_token_id, past_key_values=past_key_values, use_cache=True
            )

        logits = outputs[0]
        past_key_values = outputs[1]

        print(f"decoder kv cache {past_key_values[0][0].shape}")

        return logits, past_key_values

    def generate(prompt, tokenizer, num_tokens=10):
        inputs = tokenize_input(prompt, tokenizer)
        input_ids = inputs["input_ids"].to(device)

        logits, past_key_values = prefill_step(prompt, tokenizer)

        last_token_id = logits_to_token(logits).item()
        generated_sequence = [last_token_id]

        for i in range(num_tokens):
            logits, past_key_values = decoder_step(last_token_id, past_key_values)
            last_token_id = logits_to_token(logits).item()
            generated_sequence.append(last_token_id)
            # print(past_key_values[0][0].shape)

        print(f"{input_ids} + {generated_sequence}")
        generated_text = tokenizer.decode(generated_sequence)
        print(f"{prompt} {generated_text}")
        return generated_text, last_token_id, past_key_values

    inputs_no_pad, pad_to = get_pad_to(prompt, tokenizer)
    sequence_len = len(inputs_no_pad.input_ids[0])
    generate(prompt, tokenizer, num_tokens=pad_to - sequence_len - 1 + global_sliding)


def get_pad_to(prompt, tokenizer):
    supported_sizes = [9, 20, 30, 40]
    inputs_no_pad = tokenizer(prompt, return_tensors="pt")
    sequence_len = len(inputs_no_pad.input_ids[0])
    pad_to = min(num for num in supported_sizes if num > sequence_len)

    return inputs_no_pad, pad_to


def pad(inputs_no_pad, pad_to, tokenizer):
    padding_tok = tokenizer.eos_token
    padding_tok_id = tokenizer(padding_tok, return_tensors="pt").input_ids[0][1].item()

    sequence_len = len(inputs_no_pad.input_ids[0])

    input_ids_pad = []
    input_mask_pad = []
    for _ in range(pad_to - sequence_len):
        input_ids_pad.append(padding_tok_id)
        input_mask_pad.append(0)

    for token in inputs_no_pad.input_ids[0].tolist():
        input_ids_pad.append(token)
        input_mask_pad.append(1)

    # print(f"input_ids_pad: {input_ids_pad}")

    return torch.tensor([input_ids_pad]), torch.tensor([input_mask_pad])


def inference_cache_slice(prompt, tokenizer):

    def prefill(input_ids, attention_mask):
        with torch.no_grad():
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
            )

        logits = outputs[0]
        kv_cache = outputs[1]

        last_token_id = logits_to_token(logits)

        return last_token_id, kv_cache

    def slice_kv_cache(original_sequence_len, kv_cache):
        sliced_kv_cache = []
        if sequence_len > pad_to:
            index_to_remove = original_sequence_len
            print(f"removing kv-cache index: {index_to_remove}")
            for layer in kv_cache:
                k = layer[0]
                v = layer[1]
                sliced_k = torch.cat(
                    (k[:, :, :index_to_remove, :], k[:, :, index_to_remove + 1 :, :]),
                    dim=2,
                )
                sliced_v = torch.cat(
                    (v[:, :, :index_to_remove, :], v[:, :, index_to_remove + 1 :, :]),
                    dim=2,
                )
                sliced_kv_cache.append((sliced_k, sliced_v))

        else:
            print(f"removing kv-cache index: 0")
            for layer in kv_cache:
                k = layer[0]
                v = layer[1]
                sliced_k = k[:, :, 1:, :]
                sliced_v = v[:, :, 1:, :]
                sliced_kv_cache.append((sliced_k, sliced_v))

        return tuple(sliced_kv_cache)

    def decoder(original_sequence_len, last_token_id, kv_cache):
        last_token_id = torch.tensor([[last_token_id]], device=device)

        with torch.no_grad():
            outputs = model(last_token_id, past_key_values=kv_cache)

        logits = outputs[0]
        last_token_id = logits_to_token(logits)

        kv_cache = outputs[1]
        kv_cache = slice_kv_cache(original_sequence_len, kv_cache)

        # print(f'{kv_cache[0][0].shape} - {last_token_id.item()}')

        return last_token_id, kv_cache

    inputs_no_pad, pad_to = get_pad_to(prompt, tokenizer)
    input_ids_pad, input_mask_pad = pad(inputs_no_pad, pad_to, tokenizer)

    sequence_len = len(inputs_no_pad.input_ids[0])
    original_sequence_len = len(inputs_no_pad.input_ids[0])

    generated = []
    last_token_id, kv_cache = prefill(input_ids_pad, input_mask_pad)

    generated.append(last_token_id.item())
    sequence_len += 1
    print(f"{sequence_len} / {pad_to}:  {kv_cache[0][0].shape}")

    # slicing decoder
    for _ in range(pad_to - sequence_len + global_sliding):
        last_token_id, kv_cache = decoder(
            original_sequence_len, last_token_id, kv_cache
        )

        generated.append(last_token_id.item())
        sequence_len += 1
        print(f"{sequence_len} / {pad_to}:  {kv_cache[0][0].shape}")

    # pad_to = min(num for num in supported_sizes if num > sequence_len)
    # padding_kv = torch.zeros(kv_cache[0][0][:,:,1,:].shape)
    # padding_kv = padding_kv.unsqueeze(2)

    # padded_kv_cache = []
    # for layer in kv_cache:
    #     k = layer[0]
    #     v = layer[1]
    #     for _ in range(5):
    #         k = torch.cat((padding_kv, k), dim=2)
    #         v = torch.cat((padding_kv, k), dim=2)
    #     padded_kv_cache.append( (k, v) )

    # kv_cache = tuple(padded_kv_cache)

    # for _ in range (pad_to - sequence_len):
    #     last_token_id, kv_cache = decoder(last_token_id, kv_cache)

    #     generated.append(last_token_id.item())
    #     sequence_len += 1
    #     print(kv_cache[0][0].shape)
    #     print(f'{sequence_len} / {pad_to}\n')

    print(f"{inputs_no_pad.input_ids[0]} + {generated}")
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    print(f"{prompt} {generated_text}")


model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

device = torch.device("cpu")
model = model.to(device)

global_sliding = 20
global_sample = False

prompt = "My favorite music is "
# prompt = "The future of AI is "
# prompt = "The first president of "
print("inference_generate")
inference_generate(model, tokenizer, prompt)
print()

print("inference_prefill_decoder")
inference_prefill_decoder(model, tokenizer, prompt)
print()

print("inference_cache_slice")
inference_cache_slice(prompt, tokenizer)
print()
