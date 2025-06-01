
from lightcode import main
from lightcode import hardware
from lightcode import models

import json
import matplotlib.pyplot as plt

optimization = "time"
# optimization = "energy"
# optimization = "always_phu"

use_cpu = True
use_phu = True
use_gpu = False

CPU_AVERAGE_CLOCK = 3.208 * 10**9  # 60**9, 6
PHU_MIN_CLOCK = 9.7 * 10**9  # 100**9, 10 Ghz
GPU_FP32_CLOCK = 1.98 * 10**9  # 1.98 GHz

local_hardware = []
hardware.Hardware._hardware_reset()

if use_cpu:
    local_hardware.append(hardware.CPU(CPU_AVERAGE_CLOCK, 1))

if use_phu:
    local_hardware.append(hardware.PHU(PHU_MIN_CLOCK, 1, 20))

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

ans = main.graph_search(
    models.lenet,
    optimization,
    available_hardware,
    moc_sequence_length=1400,
    profiles=True,
    colect_data=True,
)

print(ans)
