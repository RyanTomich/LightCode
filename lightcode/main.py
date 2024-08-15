"""
Entry to program
run using conda (schedule)
"""

import json
import time
from tqdm import tqdm

import psutil

import hardware as hw
import graph_transformations as gt
import stacked_graph as sg
import testing as test
import data_collection as dc
import code_generation as cg


def forward(
    relay_path,
    optimization,
    available_hardware,
    profiles=True,
    get_step_times=True,
    data_collection=False,
):
    with open(relay_path, encoding="utf-8") as json_file:
        raw_json = json.load(json_file)  # returns json file as dict
        # print("... Json loaded ...")

    WEIGHT_VARIABLE = optimization

    graph = sg.StackGraph(raw_json=raw_json, weight_variable=WEIGHT_VARIABLE)

    # print(graph.stack_list[69])
    # slice_2 = {1027, 1288, 1549, 1810, 2071, 2332, 157, 2593, 418, 679, 940, 1201, 1462, 1723, 1984, 2245, 70, 2506, 331, 2767, 592, 853, 1114, 1375, 1636, 1897, 2158, 2419, 244, 2680, 505, 766}
    # for stack in graph.stack_list:
    #     if any(a in slice_2 for a in stack.parents):
    #         pass
    #         # print(stack.tvm_func)
    #         # print(stack.stack_id)
    # print(f'{slice_2=}')

    stacked_subgraphs = list(
        gt.graph_partition(graph, weight_variable=WEIGHT_VARIABLE)
    )
    flat_subgraphs = gt.select_nodes(
        stacked_subgraphs, weight_variable=WEIGHT_VARIABLE
    )
    expanded_flat_subgraphs = gt.expand_nodes(flat_subgraphs)
    scheduled_flat_graph, end_time, break_points = gt.schdeule_nodes(
        graph, expanded_flat_subgraphs, available_hardware
    )
    schedule_df = scheduled_flat_graph.create_schedule_data()

    # cg.code_gen(scheduled_flat_graph)

    print("---------- INFO ----------")
    print(f"{WEIGHT_VARIABLE=}")
    dc.get_photonic(flat_subgraphs)
    print(
        dc.get_all_algorithms(flat_subgraphs).symmetric_difference(
            dc.get_all_algorithms(scheduled_flat_graph)
        )
    )

    print(f"Makespan: {end_time} s")
    print(f"Number of Nodes: {len(scheduled_flat_graph.node_list)}")

    if profiles:
        dram, delta_dram, sram, delta_sram = dc.get_memory_profile(scheduled_flat_graph)
        print(f"Net DRAM: {dram[-1][1]} bits")
        print(f"Net SRAM: {sram[-1][1]} bits")
        energy_data, delta_energy, total_energy = dc.get_energy_profile(
            scheduled_flat_graph
        )
        print(f"Total Energy Consumption: {total_energy} pico-joules")
        print(
            f"time_distrabution {dc.get_time_profile(scheduled_flat_graph)} compute seconds "
        )

    print("---------- ---- ----------")

    if data_collection:
        dense_time, add_time = dc.get_addmm(scheduled_flat_graph)


def debug_forward(
    relay_path,
    optimization,
    available_hardware,
    profiles=True,
    get_step_times=True,
    config=None,
):

    # progress bar
    def mark_time():
        if get_step_times is True:
            times.append(time.time())
        progress_bar.update(1)

    times = []

    ops = [
        "open",
        "oringal_graph",
        "stacked_subgraphs",
        "flat_subgraphs",
        "expanded_flat_subgraphs",
        "schdeule_nodes",
        "create_schedule_data",
        "schedule_validate",
        "get_memory_profile",
        "get_energy_profile",
    ]

    if not profiles:
        ops.remove("get_memory_profile")
        ops.remove("get_energy_profile")

    progress_bar = tqdm(total=len(ops), desc="Progress", unit="operation")

    # Calculation
    mark_time()
    with open(relay_path, encoding="utf-8") as json_file:
        raw_json = json.load(json_file)  # returns json file as dict
        # print("... Json loaded ...")
    mark_time()

    WEIGHT_VARIABLE = optimization

    graph = sg.StackGraph(raw_json=raw_json, weight_variable=WEIGHT_VARIABLE)

    mark_time()
    stacked_subgraphs = list(gt.graph_partition(graph))
    mark_time()
    flat_subgraphs = gt.select_nodes(
        stacked_subgraphs, weight_variable=WEIGHT_VARIABLE, config=config
    )
    mark_time()
    expanded_flat_subgraphs = gt.expand_nodes(flat_subgraphs)
    mark_time()
    scheduled_flat_graph, end_time, break_points = gt.schdeule_nodes(
        graph, expanded_flat_subgraphs, available_hardware
    )
    mark_time()

    # print(scheduled_flat_graph.get_node_obj(13))

    schedule_df = scheduled_flat_graph.create_schedule_data(write=True)

    # schedule_df['difference'] = schedule_df['end'] - schedule_df['start']
    # print(schedule_df.loc[schedule_df['difference'].idxmax()])

    mark_time()
    stagnent_time = test.schedule_validate(schedule_df)
    mark_time()

    print("---------- INFO ----------")
    print(f"{WEIGHT_VARIABLE=}")
    dc.get_photonic(flat_subgraphs)
    print(
        dc.get_all_algorithms(flat_subgraphs).symmetric_difference(
            dc.get_all_algorithms(scheduled_flat_graph)
        )
    )

    print(f"Makespan: {end_time} s")
    print(f"Number of Nodes: {len(scheduled_flat_graph.node_list)}")

    if profiles:
        dram, delta_dram, sram, delta_sram = dc.get_memory_profile(scheduled_flat_graph)
        print(f"Net DRAM: {dram[-1][1]} bits")
        print(f"Net SRAM: {sram[-1][1]} bits")
        mark_time()

        energy_data, delta_energy, total_energy = dc.get_energy_profile(
            scheduled_flat_graph
        )
        print(f"Total Energy Consumption: {total_energy} pico-joules")

        mark_time()
        print(
            f"time_distrabution {dc.get_time_profile(scheduled_flat_graph)} compute seconds "
        )
        mark_time()

    if get_step_times:
        time_taken = {}
        print(len(times))
        print(len(ops))
        assert len(times) - 1 == len(ops)
        for i in range(len(ops)):
            time_taken[ops[i]] = times[i + 1] - times[i]
        print(time_taken)

    print("---------- ---- ----------")

    dense_time, add_time = dc.get_addmm(scheduled_flat_graph)


if __name__ == "__main__":  # import guard

    # relay_path = "models/gpt2_graph.json"
    # relay_path = "models/Llama-2-7b-hf_graph.json"
    relay_path = "models/opt0_Llama-2-7b-hf_graph.json"

    optimization = "time"
    # optimization = "energy"

    # cpu_freq = psutil.cpu_freq()
    # print(cpu_freq)
    # print(f"CPU Frequency: {cpu_freq.current} MHz")

    CPU_MAX_CLOCK = 5.0875 * 10**9  # 5.0875 e+9 5Ghz
    CPU_AVERAGE_CLOCK = 3.208 * 10**9  # 60**9, 6
    PHU_MIN_CLOCK = 10 * 10**9  # 100**9, 10 Ghz

    hardware = []
    hw.Hardware._hardware_reset()
    # hardware.append(hw.CPU(CPU_MAX_CLOCK, 1))
    hardware.append(hw.CPU(CPU_AVERAGE_CLOCK, 1))
    # hardware.append(hw.PHU(PHU_MIN_CLOCK, 1, 20))

    # available_hardware = hw.initilize_hardware([hw.CPU(14792899408, 1)])
    available_hardware = hw.initilize_hardware(hardware)

    forward(
        relay_path,
        optimization,
        available_hardware,
        profiles=True,
        get_step_times=False,
        data_collection=True,
    )
