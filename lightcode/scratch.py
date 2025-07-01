class GPU():
    def __init__(
        self,
        clock_speed,
        GPC,
        TPC_per_GPC,
        SM_per_TPC,
        fp32_CUDA_cores_per_SM,
        TC_per_SM,
    ):
        self.clock_speed = clock_speed
        self.mac_energy = GPU_MAC
        self.GPC = GPC  # Graphical Processing Clusters
        self.TPC_per_GPC = (
            TPC_per_GPC  # Texture Processing Clusters/Graphical Processing Cluster
        )
        self.SM_per_TPC = (
            SM_per_TPC  # Streaming multiprocessors / Texture Processing Cluster
        )
        self.fp32_CUDA_cores_per_SM = (
            fp32_CUDA_cores_per_SM  # fp32_CUDA_cores / Streaming multiprocessor
        )
        self.TC_per_SM = TC_per_SM  # Tensor Cores / Streaming multiprocessor
        self.num_cores = (
            SM_per_TPC * TPC_per_GPC * GPC
        )  # number of SM as "num_core" equivilents
        self.FLOP_per_cycle_per_tensor_core = self.get_FLOP_per_cycle_per_tensor_core(
            494.7
        )
    def get_FLOP_per_cycle_per_tensor_core(self, TFLOPS_on_tensor_core):
        tot_tensor_cores = self.num_cores * self.TC_per_SM
        FLOPS_per_tensor_core = (
            TFLOPS_on_tensor_core / tot_tensor_cores * 1_000_000_000_000
        )
        FLOP_per_cycle_per_tensor_core = FLOPS_per_tensor_core / self.clock_speed
        return FLOP_per_cycle_per_tensor_core


GPU_FP32_CLOCK = 1.98 * 10**9  # 1.98 GHz
PICO_JOULE = 10**-12
GPU_MAC = 0.07 * PICO_JOULE
GPC = 8  # Graphical Processing Clusters
TPC_per_GPC = 9  # Texture Processing Clusters/Graphical Processing Cluster
SM_per_TPC = 2  # Streaming multiprocessors / Texture Processing Cluster
fp32_CUDA_cores_per_SM = 128  # fp32_CUDA_cores / Streaming multiprocessor
TC_per_SM = 4  # Tensor Cores / Streaming multiprocessor
g = GPU(
    GPU_FP32_CLOCK,
    GPC,
    TPC_per_GPC,
    SM_per_TPC,
    fp32_CUDA_cores_per_SM,
    TC_per_SM,
)

print(g.FLOP_per_cycle_per_tensor_core)

