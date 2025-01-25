"""
Entry to program
run using conda (schedule)
"""

import psutil

from lightcode import hardware
from lightcode import graph_transformations
from lightcode import stack_graph
from lightcode import validation
from lightcode import data_collection
from lightcode import models


def graph_search(
    model,
    optimization,
    available_hardware,
    moc_sequence_length,
    profiles=True,
    colect_data=False,
):
    graph = stack_graph.StackGraph(
        model=model,
        weight_variable=optimization,
        moc_sequence_length=moc_sequence_length,
    )
    stacked_subgraphs = list(
        graph_transformations.graph_partition(graph, weight_variable=optimization)
    )
    flat_subgraphs = graph_transformations.pathfinding_node_selection(
        stacked_subgraphs, weight_variable=optimization
    )
    expanded_flat_subgraphs = graph_transformations.expand_nodes(flat_subgraphs)
    scheduled_flat_graph, end_time, break_points = graph_transformations.schdeule_nodes(
        graph, expanded_flat_subgraphs, available_hardware
    )
    schedule_df = scheduled_flat_graph.create_schedule_data()
    validation.graph_validate(scheduled_flat_graph)
    # cg.code_gen(scheduled_flat_graph)

    ret = {
        "moc_sequence_length": moc_sequence_length,
        "Makespan": end_time,
        "num_nodes": len(scheduled_flat_graph.node_list),
    }
    if profiles:
        dram, delta_dram, sram, delta_sram = data_collection.get_memory_profile(
            scheduled_flat_graph
        )
        energy_data, delta_energy, total_energy = data_collection.get_energy_profile(
            scheduled_flat_graph
        )
        ret["total_energy"] = total_energy

    if colect_data:
        selected = data_collection.get_photonic(flat_subgraphs)
        ret["num_photonic"] = selected[0]
        ret["posiable_photonic"] = selected[1]

    return ret


def threshold_search(model, optimization, available_hardware):
    graph = stack_graph.StackGraph(model=model, weight_variable=optimization)
    node_thresholds = graph_transformations.threshold_nodes(
        model, graph, weight_variable=optimization
    )
    thresholds = {}
    for node, threshold in node_thresholds.items():
        thresholds[threshold] = thresholds.get(threshold, 0) + 1
    return thresholds


if __name__ == "__main__":  # import guard

    # optimization = "time"
    optimization = "energy"
    # optimization = "always_phu"

    # cpu_freq = psutil.cpu_freq()
    # print(cpu_freq)
    # print(f"CPU Frequency: {cpu_freq.current} MHz")

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
            GPU_FP32_CLOCK,
            GPC,
            TPC_per_GPC,
            SM_per_TPC,
            fp32_CUDA_cores_per_SM,
            TC_per_SM,
        )
    )

    available_hardware = hardware.initilize_hardware(local_hardware)

    ans = graph_search(
        models.gpt2_prefill,
        optimization,
        available_hardware,
        moc_sequence_length=1400,
        profiles=True,
        colect_data=True,
    )

    # thresholds = threshold_search(
    #     models.gpt2_prefill,
    #     optimization,
    #     available_hardware,
    # )

    print(ans)
    # print(thresholds)
