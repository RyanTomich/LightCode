'''
Model instance with Relay Graph, JSON link, sequence length ran at.
'''
import json

class Model():
    def __init__(self, model, stage, relay_path, sequence_length):
        self.model = model
        self.stage = stage
        self.relay_path = relay_path
        self.sequence_length = sequence_length

    def get_raw_json(self):
        with open(self.relay_path, encoding="utf-8") as json_file:
            return json.load(json_file)


relay_path = "models/gpt2_prefill_graph.json"
# relay_path = "models/gpt2_decoder_graph.json"
# relay_path = "models/gpt2_graph.json"
# relay_path = "models/Llama-2-7b-hf_graph.json"
# relay_path = "models/opt0_Llama-2-7b-hf_graph.json"
# relay_path = "models/llama_2_7b_decoder_graph.json" # 10

# relay_path = "models/len_comparison/gpt2_decoder_graph_5.json"
# relay_path = "models/len_comparison/gpt2_decoder_graph_6.json"
# relay_path = "models/len_comparison/gpt2_prefill_graph_5.json"
# relay_path = "models/len_comparison/gpt2_prefill_graph_6.json"


# relay_path = "models/gpt2_graph.json" # 5
# relay_path = "models/gpt2_prefill_graph.json"
# relay_path = "models/gpt2_decoder_graph.json" # 6
# relay_path = "models/Llama-2-7b-hf_graph.json" # 6
# relay_path = "models/opt0_Llama-2-7b-hf_graph.json" # 6
# relay_path = "models/llama_2_7b_decoder_graph.json" # 11

llama_prefill = Model(
    model = 'Llama-2-7b-hf',
    stage = 'prefill',
    relay_path = "models/opt0_Llama-2-7b-hf_graph.json",
    sequence_length = 6,
)

llama_decoder  = Model(
    model = 'Llama-2-7b-hf',
    stage = 'decoder',
    relay_path = "models/llama_2_7b_decoder_graph",
    sequence_length = 11,
)

gpt2_prefill = Model(
    model = 'gpt2',
    stage = 'prefill',
    relay_path = "models/gpt2_prefill_graph.json",
    sequence_length = 5,
)

gpt2_decoder = Model(
    model = 'gpt2',
    stage = 'decoder',
    relay_path = "models/gpt2_decoder_graph.json",
    sequence_length = 6,
)
