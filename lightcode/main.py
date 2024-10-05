"""
Entry to program
run using conda (schedule)
"""

import json
import psutil

import hardware as hw
import graph_transformations as gt
import stacked_graph as sg
import input_validation as validate
import data_collection as dc
import code_generation as cg


def open_json(path):
    with open(path, encoding="utf-8") as json_file:
        return json.load(json_file)


def graph_search(
    relay_path,
    optimization,
    available_hardware,
    moc_sequence_length,
    profiles=True,
    data_collection=False,
):
    raw_json = open_json(relay_path)

    WEIGHT_VARIABLE = optimization

    graph = sg.StackGraph(raw_json=raw_json, weight_variable=WEIGHT_VARIABLE, moc_sequence_length = moc_sequence_length)
    stacked_subgraphs = list(gt.graph_partition(graph, weight_variable=WEIGHT_VARIABLE))
    flat_subgraphs = gt.pathfinding_node_selection(
        stacked_subgraphs, weight_variable=WEIGHT_VARIABLE
    )
    expanded_flat_subgraphs = gt.expand_nodes(flat_subgraphs)
    scheduled_flat_graph, end_time, break_points = gt.schdeule_nodes(
        graph, expanded_flat_subgraphs, available_hardware
    )
    schedule_df = scheduled_flat_graph.create_schedule_data()

    validate.graph_validate(scheduled_flat_graph)
    cg.code_gen(scheduled_flat_graph)

    # print("---------- INFO ----------")
    # print(f"{WEIGHT_VARIABLE=}")
    # print(f"{moc_sequence_length=}")
    # dc.get_photonic(flat_subgraphs)
    # print(
    #     dc.get_all_algorithms(flat_subgraphs).symmetric_difference(
    #         dc.get_all_algorithms(scheduled_flat_graph)
    #     )
    # )

    # print(f"Makespan: {end_time} s")
    # print(f"Number of Nodes: {len(scheduled_flat_graph.node_list)}")
    total_energy = None
    if profiles:
        dram, delta_dram, sram, delta_sram = dc.get_memory_profile(scheduled_flat_graph)
        # print(f"Net DRAM: {dram[-1][1]} bits")
        # print(f"Net SRAM: {sram[-1][1]} bits")
        energy_data, delta_energy, total_energy = dc.get_energy_profile(
            scheduled_flat_graph
        )
        # print(f"Total Energy Consumption: {total_energy} pico-joules")
        # print(
            # f"time_distrabution {dc.get_time_profile(scheduled_flat_graph)} compute seconds "
        # )

    # print("---------- ---- ----------")

    return {
        "moc_sequence_length": moc_sequence_length,
        "Makespan": end_time,
        "total_energy": total_energy,
        "num_nodes": len(scheduled_flat_graph.node_list)
    }


def threshold_search(relay_path, optimization, available_hardware):
    WEIGHT_VARIABLE = optimization
    raw_json = open_json(relay_path)
    graph = sg.StackGraph(raw_json=raw_json, weight_variable=WEIGHT_VARIABLE)
    node_thresholds = gt.threshold_nodes(graph, weight_variable=WEIGHT_VARIABLE)
    thresholds = set()
    for i, v in node_thresholds.items():
        thresholds.add(v)
    print(thresholds)


if __name__ == "__main__":  # import guard

    # relay_path = "models/gpt2_prefill_graph.json"
    relay_path = "models/gpt2_decoder_graph.json"
    # relay_path = "models/gpt2_graph.json"
    # relay_path = "models/Llama-2-7b-hf_graph.json"
    # relay_path = "models/opt0_Llama-2-7b-hf_graph.json"

    optimization = "time"
    # optimization = "energy"

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

    # available_hardware = hw.initilize_hardware([hw.CPU(14792899408, 1)])
    available_hardware = hw.initilize_hardware(hardware)

    ans = graph_search(
        relay_path,
        optimization,
        available_hardware,
        moc_sequence_length = 100,
        profiles=True,
        data_collection=False,
    )

    print(ans)

    # threshold_search(
    #     relay_path,
    #     optimization,
    #     available_hardware,
    # )
