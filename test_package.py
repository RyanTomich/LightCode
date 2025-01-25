from lightcode import main
from lightcode import hardware
from lightcode import models

# optimization = "time"
optimization = "energy"
# optimization = "always_phu"

CPU_MAX_CLOCK = 5.0875 * 10**9  # 5.0875 e+9 5Ghz
CPU_AVERAGE_CLOCK = 3.208 * 10**9  # 60**9, 6
PHU_MIN_CLOCK = 9.7 * 10**9  # 100**9, 10 Ghz
GPU_FP32_CLOCK = 1.98 * 10**9  # 1.98 GHz

local_hardware = []
hardware.Hardware._hardware_reset()
# local_hardware.append(hardware.CPU(CPU_MAX_CLOCK, 1))
local_hardware.append(hardware.CPU(CPU_AVERAGE_CLOCK, 1))
# local_hardware.append(hardware.PHU(PHU_MIN_CLOCK, 1, 20))

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
    models.gpt2_prefill,
    optimization,
    available_hardware,
    moc_sequence_length=1400,
    profiles=True,
    colect_data=True,
)
print(ans)

thresholds = main.threshold_search(
    models.gpt2_prefill,
    optimization,
    available_hardware,
)
print(thresholds)
