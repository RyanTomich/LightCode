---
layout: default
title: "Results"
nav_order: 5
---
This assumes you have already done the [Installation Guide](installation.md).

Switch to the `schedule` conda envuroemnt

# Raw sim results
Complete code can be found ()

Import lightcode
```python
from lightcode import main
from lightcode import models
```

Define youre hardware
```python
CPU_AVERAGE_CLOCK = 3.208 * 10**9  # 60**9, 6
PHU_MIN_CLOCK = 9.7 * 10**9  # 100**9, 10 Ghz
GPU_FP32_CLOCK = 1.98 * 10**9  # 1.98 GHz

local_hardware = []
main.hardware.Hardware._hardware_reset()
local_hardware.append(main.hardware.CPU(CPU_AVERAGE_CLOCK, 1)) # 1 core
local_hardware.append(main.hardware.PHU(PHU_MIN_CLOCK, 1, 20)) # 1 core, 20 multiplex factor

GPC = 8  # Graphical Processing Clusters
TPC_per_GPC = 9  # Texture Processing Clusters/Graphical Processing Cluster
SM_per_TPC = 2  # Streaming multiprocessors / Texture Processing Cluster
fp32_CUDA_cores_per_SM = 128  # fp32_CUDA_cores / Streaming multiprocessor
TC_per_SM = 4  # Tensor Cores / Streaming multiprocessor
local_hardware.append(main.hardware.GPU(
        GPU_FP32_CLOCK,
        GPC,
        TPC_per_GPC,
        SM_per_TPC,
        fp32_CUDA_cores_per_SM,
        TC_per_SM,
    )
)

available_hardware = main.hardware.initilize_hardware(local_hardware)
```

- NOTE: Definig a hardware type (`hardwar.CPU`, `hardware.PHU`, etc.) adds that hardware to the sims resources. The sim will expect that hardware to be part of the initilization. Before anohter sim with different hardware paramatrs, you need to run the `hardware.Hardware._hardware_reset()` and redefine and reinitilize the new hardware.

Define optimization paramater and run graph search
```python
optimization = "time"
# optimization = "energy"
# optimization = "always_phu"

ans = main.graph_search(
    models.gpt2_prefill,
    optimization,
    available_hardware,
    moc_sequence_length=50,
    profiles=True,
    colect_data=True,
)
```
{'moc_sequence_length': 50, 'Makespan': 0.04984, 'num_nodes': 1518, 'total_energy': 43079811496.0, 'num_photonic': 73, 'posiable_photonic': 73}

# Plots

[^1]: In this case, 73 out of 73 possible photonic operations were selected. This indicates that:
The model has 73 total operations that can be computed by a photonic processor (matrix multiplication).
When optimizing for time, the graph search algorithm determined that all 73 operations should be computed using photonics.
If the selection ratio were lower (e.g., 25/73), it would imply that only a subset of operations would benefit from being executed on photonic hardware, likely due to differences in operation sizes (e.g., larger matrix multiplications).
