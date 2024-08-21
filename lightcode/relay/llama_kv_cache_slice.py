import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Cache


def prefill(input_ids, attention_mask):
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
        )

    logits = outputs[0]
    kv_cache = outputs[1]

    last_token_id = torch.argmax(logits[:, -1, :], dim=-1)

    return last_token_id, kv_cache


def slice_kv_cache(kv_cache):
    sliced_kv_cache = []
    for layer in kv_cache:
        k = layer[0]
        v = layer[1]
        sliced_k = k[:, :, 1:, :]
        sliced_v = v[:, :, 1:, :]
        sliced_kv_cache.append((sliced_k, sliced_v))

    return tuple(sliced_kv_cache)


def decoder(last_token_id, kv_cache):
    last_token_id = torch.tensor([[last_token_id]], device=device)

    with torch.no_grad():
        outputs = model(last_token_id, past_key_values=kv_cache)

    logits = outputs[0]
    last_token_id = torch.argmax(logits[:, -1, :], dim=-1)

    kv_cache = outputs[1]
    kv_cache = slice_kv_cache(kv_cache)

    # print(f'{kv_cache[0][0].shape} - {last_token_id.item()}')

    return last_token_id, kv_cache


model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

padding_tok = tokenizer.eos_token
padding_tok_id = tokenizer(padding_tok, return_tensors="pt").input_ids[0][1].item()


model.eval()

device = torch.device("cpu")
model = model.to(device)

# supported_sizes = [10,20,30,40]
supported_sizes = [15, 20, 30, 40]
# supported_sizes = [20,30,40]
# supported_sizes = [30,40]

# prompt = "My favorite music is "
prompt = "The future of AI is "

inputs_no_pad = tokenizer(prompt, return_tensors="pt")
sequence_len = len(inputs_no_pad.input_ids[0])

print(f"input_ids: {inputs_no_pad.input_ids[0].tolist()}")

pad_to = min(num for num in supported_sizes if num > sequence_len)

input_ids_pad = []
input_mask_pad = []
for _ in range(pad_to - sequence_len):
    input_ids_pad.append(padding_tok_id)
    input_mask_pad.append(0)

for token in inputs_no_pad.input_ids[0].tolist():
    input_ids_pad.append(token)
    input_mask_pad.append(1)

print(f"input_ids_pad: {input_ids_pad}")

input_ids_pad = torch.tensor([input_ids_pad])
input_mask_pad = torch.tensor([input_mask_pad])

generated = []

last_token_id, kv_cache = prefill(input_ids_pad, input_mask_pad)

generated.append(last_token_id.item())
sequence_len += 1
print(kv_cache[0][0].shape)
print(f"{sequence_len} / {pad_to}\n")

for _ in range(pad_to - sequence_len):
    last_token_id, kv_cache = decoder(last_token_id, kv_cache)

    generated.append(last_token_id.item())
    sequence_len += 1
    print(kv_cache[0][0].shape)
    print(f"{sequence_len} / {pad_to}\n")

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
