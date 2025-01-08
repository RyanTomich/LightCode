"""
Entry to program
run using conda (schedule)
"""

import psutil

import hardware as hw
import graph_transformations as gt
import stacked_graph as sg
import input_validation as validate
import data_collection as dc
import code_generation as cg
import models as models


def graph_search(
    model,
    optimization,
    available_hardware,
    moc_sequence_length,
    profiles=True,
    data_collection=False,
):
    graph = sg.StackGraph(
        model=model,
        weight_variable=optimization,
        moc_sequence_length=moc_sequence_length,
    )
    stacked_subgraphs = list(gt.graph_partition(graph, weight_variable=optimization))
    flat_subgraphs = gt.pathfinding_node_selection(
        stacked_subgraphs, weight_variable=optimization
    )
    expanded_flat_subgraphs = gt.expand_nodes(flat_subgraphs)
    scheduled_flat_graph, end_time, break_points = gt.schdeule_nodes(
        graph, expanded_flat_subgraphs, available_hardware
    )
    schedule_df = scheduled_flat_graph.create_schedule_data()
    validate.graph_validate(scheduled_flat_graph)
    # cg.code_gen(scheduled_flat_graph)

    ret = {
        "moc_sequence_length": moc_sequence_length,
        "Makespan": end_time,
        "num_nodes": len(scheduled_flat_graph.node_list),
    }
    if profiles:
        dram, delta_dram, sram, delta_sram = dc.get_memory_profile(scheduled_flat_graph)
        energy_data, delta_energy, total_energy = dc.get_energy_profile(
            scheduled_flat_graph
        )
        ret["total_energy"] = total_energy

    if data_collection:
        selected = dc.get_photonic(flat_subgraphs)
        ret["num_photonic"] = selected[0]
        ret["posiable_photonic"] = selected[1]

    return ret


def threshold_search(model, optimization, available_hardware):
    graph = sg.StackGraph(model=model, weight_variable=optimization)
    node_thresholds = gt.threshold_nodes(model, graph, weight_variable=optimization)
    thresholds = {}
    for node, threshold in node_thresholds.items():
        thresholds[threshold] = thresholds.get(threshold, 0) + 1
    return thresholds


if __name__ == "__main__":  # import guard

    # optimization = "time"
    # optimization = "energy"
    optimization = "always_phu"

    # cpu_freq = psutil.cpu_freq()
    # print(cpu_freq)
    # print(f"CPU Frequency: {cpu_freq.current} MHz")

    CPU_MAX_CLOCK = 5.0875 * 10**9  # 5.0875 e+9 5Ghz
    CPU_AVERAGE_CLOCK = 3.208 * 10**9  # 60**9, 6
    PHU_MIN_CLOCK = 9.7 * 10**9  # 100**9, 10 Ghz

    hardware = []
    hw.Hardware._hardware_reset()
    # hardware.append(hw.CPU(CPU_MAX_CLOCK, 1))
    hardware.append(hw.CPU(CPU_AVERAGE_CLOCK, 1))
    hardware.append(hw.PHU(PHU_MIN_CLOCK, 1, 20))

    available_hardware = hw.initilize_hardware(hardware)

    ans = graph_search(
        models.gpt2_prefill,
        optimization,
        available_hardware,
        moc_sequence_length=1400,
        profiles=True,
        data_collection=True,
    )

    thresholds = threshold_search(
        models.gpt2_prefill,
        optimization,
        available_hardware,
    )

    print(ans)
    print(thresholds)
