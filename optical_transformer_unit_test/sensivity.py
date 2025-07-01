from lightcode import main
from lightcode import hardware
from lightcode import models
import json

PICO_JOULE = 10**-12
BITS_PER_NUM = 8

# Parameters defined directly in hardware.py
hardware_parameters = {
    "SRAM_RW_COST",
    "MEMORY_CLOCK",
    "DRAM_RW_COST",
    "LOCAL_RW_COST",
    "PHU_MAC",
    "GPU_MAC",
    "DAC_POWER",
    "ADC_POWER",
}

# Parameters NOT in hardware.py — defined manually below
manual_parameters = {
    "PHU_MIN_CLOCK": 9.7e9,
    "GPU_FP32_CLOCK": 1.98e9,
}

varying_parameters = list(manual_parameters.keys()| hardware_parameters)


model_names = [("gpt2_prefill", 1024), ("llama_prefill", 4096)]
optimizations = ["time", "energy"]


for var_param in varying_parameters:

    if var_param in hardware_parameters:
        base_val = getattr(hardware, var_param)
    elif var_param in manual_parameters:
        base_val = manual_parameters[var_param]
    else:
        raise ValueError(f"Unknown parameter: {var_param}")
    param_vals = [base_val * scale for scale in [0.9, 0.95, 1.0, 1.05, 1.1]]

    for model_name, seq_len in model_names:
        for optimization in optimizations:

            results = {}

            for param_val in param_vals:
                PHU_MIN_CLOCK = 9.7e9
                GPU_FP32_CLOCK = 1.98e9

                if var_param in hardware_parameters:
                    hardware.reset_param(var_param, param_val)
                else:
                    if var_param == "PHU_MIN_CLOCK":
                        PHU_MIN_CLOCK = param_val
                    elif var_param == "GPU_FP32_CLOCK":
                        GPU_FP32_CLOCK = param_val

                model = getattr(models, model_name)

                use_phu = True
                use_gpu = True

                local_hardware = []
                hardware.Hardware._hardware_reset()

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

                ans = main.graph_search(
                    model,
                    optimization,
                    available_hardware,
                    moc_sequence_length=seq_len,
                    profiles=True,
                    colect_data=True,
                )
                results[param_val] = ans

            # print(results)

            import os
            save_name = f"{model_name}_{seq_len}_{optimization}_{var_param}"
            print(save_name)

            json_path = f'sensitivity_results_max.json'

            # Load existing results
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    json_file = json.load(f)
            else:
                json_file = {}

            json_file[save_name] = results

            with open(json_path, 'w') as f:
                json.dump(json_file, f, indent=2)

    if var_param in hardware_parameters:
        hardware.reset_param(var_param, base_val)
