import json

# relay_path ='../lightcode/models/gpt2_graph.json'
relay_path = "../lightcode/models/Llama-2-7b-hf_graph.json"

with open(relay_path, encoding="utf-8") as json_file:
    raw_json = json.load(json_file)

funcs = set()
for node in raw_json["nodes"]:
    if "attrs" in node:
        funcs.add(node["attrs"]["func_name"])

print(funcs)
