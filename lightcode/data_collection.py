import hardware as hw


def get_photonic(subgraphs):
    """conpairs how often photonic was selected

    Args:
        subgraphs (list of Graph): list of flat graphs before expansion
    """
    total = 0
    selected = 0

    for subgraph in subgraphs:
        for node in subgraph.node_list:

            for alg, algorithm_obj in hw.Hardware.algs.items():
                if node.stack.opp == algorithm_obj.opp and "phu" in alg:
                    total += 1
                    break

            if "phu" in node.algorithm:
                selected += 1

    print(f"Photonic Selected: {selected} / {total}")


def get_memory_profile(graph):
    """time datapoints of memory changes in bits

    Args:
        graph (Graph):

    Returns:
        lsit of tuples: representing total and change in
    """
    delta_dram = []
    delta_sram = []
    dram = []  # (time, bits)
    sram = []  # (time, bits)
    dram_total = 0
    sram_total = 0

    outdegree = {
        idx: sum(1 for i in row if i is not None)
        for idx, row in enumerate(graph.adj_matrix)
    }

    dram.append((0, 0))
    sorted_stack_list = sorted(graph.node_list, key=lambda x: x.start_time)

    for node_obj in sorted_stack_list:
        in_size = sum(hw.ten_elm(x) for x in node_obj.input_shapes)
        out_size = sum(hw.ten_elm(x) for x in node_obj.output_shapes)
        assert out_size >= 0

        if node_obj.stack_id in graph.in_nodes:  # D -> S
            dram_total -= out_size
            dram.append((node_obj.start_time, dram_total))
            delta_dram.append((node_obj.start_time, -out_size))

            sram_total += out_size
            sram.append(
                (
                    node_obj.start_time + node_obj.time_cost,
                    sram_total,
                )
            )
            delta_sram.append(
                (
                    node_obj.start_time + node_obj.time_cost,
                    out_size,
                )
            )

        elif node_obj.stack_id in graph.out_nodes:  # S -> D
            sram_total -= in_size
            sram.append((node_obj.start_time, sram_total))
            delta_sram.append((node_obj.start_time, -in_size))

            dram_total += out_size
            dram.append(
                (
                    node_obj.start_time + node_obj.time_cost,
                    dram_total,
                )
            )
            delta_dram.append(
                (
                    node_obj.start_time + node_obj.time_cost,
                    out_size,
                )
            )

        else:
            for parent in node_obj.parents:
                parent_obj = graph.get_node_obj(parent)
                outdegree[graph.id_to_idx[parent_obj.stack_id]] -= 1
                # once all children are satisfied, we remove data from SRAM
                if outdegree[graph.id_to_idx[parent_obj.stack_id]] == 0:
                    size = sum(hw.ten_elm(x) for x in parent_obj.output_shapes)
                    sram_total -= size
                    sram.append((node_obj.start_time, sram_total))
                    delta_sram.append((node_obj.start_time, -size))

            sram_total += out_size
            sram.append(
                (
                    node_obj.start_time + node_obj.time_cost,
                    sram_total,
                )
            )
            delta_sram.append(
                (
                    node_obj.start_time + node_obj.time_cost,
                    out_size,
                )
            )

    # print(f"{dram_total=}")
    # print(f"{sram_total=}")
    print("... Memory profile made ...") if hw.DEBUG_PRINT else None
    return dram, delta_dram, sram, delta_sram


def get_all_algorithms(graph):
    if not isinstance(graph, list):
        graph = [graph]

    algorithms = set()
    for subgraph in graph:
        for node in subgraph.node_list:
            algorithms.add(node.algorithm)

    # print(algorithms)
    return algorithms


def get_energy_profile(graph):
    """creates list of data points for energy usage
    Makes the assumption that all energy usage happens at the beginning when in
    reality, power is continuously used over the time of each transfer or computation.

    Args:
        graph (Graph): scheduled computational graph
    """
    delta_energy = []  # (time, energy_usage)

    # Node Energy
    for node in graph.node_list:
        delta_energy.append((node.start_time, node.energy_cost))

    # Edge energy
    for row_num in range(len(graph.adj_matrix)):
        for col_num in range(len(graph.adj_matrix)):
            if graph.adj_matrix[row_num][col_num] is None:
                continue

            start_node = graph.node_list[row_num]
            end_node = graph.node_list[col_num]

            energy_change = hw.get_edge_val(graph, start_node, end_node, "energy")

            delta_energy.append((start_node.start_time, energy_change))

    delta_energy.sort(key=lambda x: x[0])

    total_energy = 0
    energy_data = [(0, 0)]
    for delta in delta_energy:
        total_energy += delta[1]
        energy_data.append((delta[0], total_energy))

    return energy_data, delta_energy, round(total_energy * 1 / hw.PICO_JOULE, 1)


def get_time_profile(graph):
    time_profile = {}
    for node in graph.node_list:
        time_profile.setdefault(node.algorithm, 0)
        time_profile[node.algorithm] += node.time_cost

    return time_profile


def get_energy_per_opp(graph):
    energy_profile = {}
    for node in graph.node_list:
        energy_profile.setdefault(node.algorithm, 0)
        energy_profile[node.algorithm] += node.energy_cost

    return energy_profile


def time_energy_profiles(graph):
    time_profile = {}
    energy_profile = {}
    for node in graph.node_list:
        time_profile.setdefault(node.algorithm, 0)
        time_profile[node.algorithm] += node.time_cost

        energy_profile.setdefault(node.algorithm, 0)
        energy_profile[node.algorithm] += node.energy_cost

    return time_profile, energy_profile


def get_addmm(scheduled_flat_graph):
    dense_time = 0
    add_time = 0
    # add = 0
    # for node in scheduled_flat_graph.node_list:
    #     # if node.input_shapes == shape and node.algorithm == 'multiply':
    #     if node.algorithm == "matmul":

    #         print(node.stack.tvm_func)
    #         print(f'{node.input_shapes} -- {node.output_shapes}')
    #         print(f'{node.time_cost}')

    return dense_time, add_time
