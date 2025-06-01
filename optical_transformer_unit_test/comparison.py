#%%

from lightcode import main
from lightcode import hardware
from lightcode import models

import json
import matplotlib.pyplot as plt


#%%
# optimization = "time"
optimization = "energy"
# optimization = "always_phu"

use_cpu = False
use_phu = True
use_gpu = True

CPU_AVERAGE_CLOCK = 3.208 * 10**9  # 60**9, 6
PHU_MIN_CLOCK = 9.7 * 10**9  # 100**9, 10 Ghz
GPU_FP32_CLOCK = 1.98 * 10**9  # 1.98 GHz

local_hardware = []
hardware.Hardware._hardware_reset()

if use_cpu:
    local_hardware.append(hardware.CPU(CPU_AVERAGE_CLOCK, 1))

if use_phu:
    phu_cores = 1
    phu_multiplex = 20
    local_hardware.append(hardware.PHU(PHU_MIN_CLOCK, phu_cores, phu_multiplex))

if use_gpu:
    GPC = 8  # Graphical Processing Clusters
    TPC_per_GPC = 9  # Texture Processing Clusters/Graphical Processing Cluster
    SM_per_TPC = 2  # Streaming multiprocessors / Texture Processing Cluster
    fp32_CUDA_cores_per_SM = 128  # fp32_CUDA_cores / Streaming multiprocessor
    TC_per_SM = 4  # Tensor Cores / Streaming multiprocessor
    local_hardware.append(
        hardware.GPU(
            GPU_FP32_CLOCK, GPC, TPC_per_GPC, SM_per_TPC, fp32_CUDA_cores_per_SM, TC_per_SM
        )
    )

available_hardware = hardware.initilize_hardware(local_hardware)

# ans = main.graph_search(
#     models.gpt2_prefill,
#     optimization,
#     available_hardware,
#     moc_sequence_length=1400,
#     profiles=True,
#     colect_data=True,
# )
# print(ans)


# %%
sequence_lengths = [i  * 100 for i in range(40)]
results = {}

# Run the sweep
# model_name = "gpt2_prefill"
model_name = "llama_prefill"
model = getattr(models, model_name)
for seq_len in sequence_lengths:
    ans = main.graph_search(
        model,
        optimization,
        available_hardware,
        moc_sequence_length=seq_len,
        profiles=True,
        colect_data=True,
    )
    results[seq_len] = ans

# %%
results_with_metadata = {
    "_metadata": {
        "optimization": optimization,
        "hardware": {
            "CPU": {
                "enabled": use_cpu,
                "clock_Hz": CPU_AVERAGE_CLOCK
            },
            "PHU": {
                "enabled": use_phu,
                "clock_Hz": PHU_MIN_CLOCK,
                "num_cores": phu_cores,
                "num_multiplex": phu_multiplex
            },
            "GPU": {
                "enabled": use_gpu,
                "clock_Hz": GPU_FP32_CLOCK,
                "GPC": GPC,
                "TPC_per_GPC": TPC_per_GPC,
                "SM_per_TPC": SM_per_TPC,
                "fp32_CUDA_cores_per_SM": fp32_CUDA_cores_per_SM,
                "TC_per_SM": TC_per_SM
            }
        },
        "graph_search": {
            "model": model_name,
        }
    },
    "results": results
}

# Save results to JSON
filename = f"{model_name}_{optimization}_CPU-{use_cpu}_PHU-{use_phu}_{phu_cores}_{phu_multiplex}_GPU-{use_gpu}.json"
with open(filename, "w") as f:
    json.dump(results_with_metadata, f, indent=2)

# %%
