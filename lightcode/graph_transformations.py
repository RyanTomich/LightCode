"""
All complex graph operations and redefinitions
"""

import copy
import heapq
import numpy as np

import stacked_graph as sg
import input_validation as validate
import hardware as hw
import photonic_algorithms as pa
import data_collection as dc

node_value_selection = {
    "time": lambda node: node.time_cost,
    "energy": lambda node: node.energy_cost,
}


# region graph_partition


def graph_partition(graph, weight_variable="time"):
    """Finds the Articulation Vertices and partitions the large graph into subgraphs
    StackedGraph objects. Inclusive on both ends of range.
    graph: StackedGraph
    """
    scatter_stack = None
    for stack in graph.stack_list:
        if stack.opp == "nd":
            scatter_stack = stack
            continue

    for stack in graph.stack_list:
        if scatter_stack and (scatter_stack.stack_id in stack.parents):
            graph.residual.add(stack.stack_id)

    groups = list(graph.get_node_groups(asap=False))
    validate.group_validate(graph, groups)
    subgraphs = []

    for group in groups:

        start_stack = sg.Stack(0, set(), [[]], [[]], None, opp="start", node_stack=[])
        start_stack.node_stack.append(sg.Node("start", start_stack))

        # replace parents if not satisfied in group. handles loads and residuals
        first_stacks = []
        stacks_hit = set()
        for stack in group:
            stack_obj = graph.get_node_obj(stack)
            # first stack
            if all(parent not in group for parent in stack_obj.parents):
                stacks_hit.add(stack_obj.stack_id)
                first_stack = copy.deepcopy(stack_obj)
                first_stack.parents = {0}
                first_stacks.append(first_stack)

        subgraph_stack_list = [start_stack] + first_stacks
        for stack_id in group:
            if stack_id not in stacks_hit:
                stack_obj = graph.get_node_obj(stack_id)
                new_node = copy.deepcopy(stack_obj)
                new_node.parents = set(new_node.parents) - graph.in_nodes

                # add energy weight from in_node to nodes in stack
                memory_parent = set(stack_obj.parents) & graph.in_nodes
                if weight_variable == "energy" and len(memory_parent) > 0:
                    stack_idx = graph.id_to_idx[stack_id]
                    parent_idx = graph.id_to_idx[memory_parent.pop()]
                    edge_energy_cost = graph.adj_matrix[parent_idx][stack_idx]
                    for idx, cost in enumerate(edge_energy_cost[0]):
                        new_node.node_stack[idx].energy_cost += cost

                subgraph_stack_list.append(new_node)

        sub_graph = sg.StackGraph(
            stack_list=subgraph_stack_list,
            weight_variable=graph.weight_variable,
        )
        subgraphs.append(sub_graph)

    return subgraphs


# endregion


# region selection


def _extract_stacks(path):
    """returns set of stacks included in the path

    Args:
        path (list of tuples): [ (stack, node) ]

    Returns:
        set: stacks included in the path
    """
    return {index[0] for index in path}


def _make_aggreement_list(graph):
    """
    graph: StackedGraph Object
    returns all stack indicies with branches or merges
    """
    branches = []
    stack_list = graph.stack_list
    adj_matrix = graph.adj_matrix
    stack_opps = [stack.opp for stack in stack_list]

    # Ignores memory nodes
    for idx in range(len(stack_list)):
        row_counts = sum(
            1
            for stack_idx in range(len(stack_list))
            if adj_matrix[idx, stack_idx] is not None
            and stack_opps[stack_idx] != "memory"
        )

        col_count = sum(
            1
            for stack_idx in range(adj_matrix.shape[0])
            if adj_matrix[stack_idx, idx] is not None
            and stack_opps[stack_idx] != "memory"
        )

        # has> 1 parent or >1 child
        if row_counts > 1 or col_count > 1:
            branches.append(idx)

    return branches


def _get_aggreement(node_indexes, aggreement_stacks):
    """
    returns tuple of the nodes in each of the aggreement stacks
    node_indexes: a path
    aggreement_stacks: list of stacks that must aggreee
    """
    if not aggreement_stacks:
        return "all"

    stack_index_map = {node: i for i, node in enumerate(aggreement_stacks)}
    stack_indexes = [None] * len(aggreement_stacks)

    for node in node_indexes:
        index = stack_index_map.get(node[0])
        if index is not None:
            stack_indexes[index] = node[1]

    return tuple(stack_indexes)


def _ap_works(group_ap, new_ap):
    """checks that articulation points match

    Args:
        group_ap (set): all merge points
        new_ap (set): all types

    Returns:
        bool: match
    """
    return all(
        new_ap[idx] == node or new_ap[idx] is None or node is None
        for idx, node in enumerate(group_ap)
    )


def _add_group(
    groups, group, stack_aggreement, cur_path, stack_coverage, quick_heuristic=False
):
    """adds this path to the group of paths

    Args:
        groups (list)): all groups of paths
        group (dict): group to add to
        stack_aggreement (set): which nodes need to aggree
        cur_path (list): this path
        stack_coverage (set): Current path coverage
    """

    # Faster method by considering found paths as optimal and extending them if posiable.
    if quick_heuristic == True:
        new_ap = list(group["ap"])
        for idx in range(len(group["ap"])):
            new_aggreement = stack_aggreement[idx]
            if new_aggreement is not None:
                new_ap[idx] = new_aggreement

        group["ap"] = tuple(new_ap)
        group["paths"] += cur_path
        group["coverage_groups"].append(stack_coverage)
        group["total_coverage"].update(stack_coverage)

    # Slower method by considering and extending them, but also leaving them around for later additions
    else:
        new = False
        for idx, val in enumerate(group["ap"]):
            if (val is None) != (stack_aggreement[idx] is None) and (
                val is None or stack_aggreement[idx] is None
            ):
                new = True

        if new:  # there are None's so keep original
            new_group = copy.deepcopy(group)
            new_group["ap"] = [
                (a if a is not None else b)
                for a, b in zip(new_group["ap"], stack_aggreement)
            ]
            new_group["paths"] += cur_path
            new_group["coverage_groups"].append(stack_coverage)
            new_group["total_coverage"].update(stack_coverage)
            groups.append(new_group)

        else:  # perfect match so mutate group in place
            group["ap"] = [
                (a if a is not None else b)
                for a, b in zip(group["ap"], stack_aggreement)
            ]
            group["paths"] += cur_path
            group["coverage_groups"].append(stack_coverage)
            group["total_coverage"].update(stack_coverage)


def _select_first_occorance(group):
    stacks_seen = set()
    nodes = set()
    for node in group["paths"]:
        if node[0] not in stacks_seen:
            nodes.add(node)
            stacks_seen.add(node[0])
    return nodes


def _ending_node(cur_path, aggreement_stacks, groups, all_nodes):
    """checks if ending node has made a match. All inputs mutable

    Returns:
        set: set of working nodes if end satisfies
    """

    stack_aggreement = _get_aggreement(cur_path, aggreement_stacks)
    stack_coverage = _extract_stacks(cur_path)

    if stack_aggreement == "all" and stack_coverage == all_nodes:
        return set(cur_path)

    added = False
    for group in groups:
        if (
            _ap_works(group["ap"], stack_aggreement)  # aggreement matches
            and stack_coverage - group["total_coverage"] != {}  # adds coverage
        ):
            _add_group(
                groups,
                group,
                stack_aggreement,
                cur_path,
                stack_coverage,
                quick_heuristic=True,
            )
            added = True

            if group["total_coverage"] == all_nodes:  # group reached full coverage:
                # paths are in increasing order of time. select earliest occurrence of a stack
                return _select_first_occorance(group)

    if not added:
        groups.append(
            {
                "ap": (stack_aggreement),
                "paths": tuple(cur_path),
                "coverage_groups": [stack_coverage],
                "total_coverage": stack_coverage,
            }
        )
    return None


def _rolling_dijkstra(graph, weight_variable):
    """
    Dijkstra untill there is full coverage on a combination of aggreement stacks
    graph to optimize
    return list of (stack_idx, node_idx)
    """
    aggreement_stacks = _make_aggreement_list(graph)
    all_nodes = {i for i, v in enumerate(graph.stack_list) if v.opp != "memory"}

    que = []
    for stack_id in graph.in_nodes:
        que.append(
            (0, ((graph.id_to_idx[stack_id], 0),))
        )  # (cur_dist, ( (stack_idx), node_idx), ) )
    groups = []

    while que:
        cur_dist, cur_path = heapq.heappop(que)  # minimum
        cur_node = cur_path[-1]  # last item in path
        neighbor_stacks = graph.get_stack_neighbors(cur_node[0])

        if neighbor_stacks == []:  # ending node
            found = _ending_node(cur_path, aggreement_stacks, groups, all_nodes)
            if found:
                return found

        for neighbor in neighbor_stacks:
            stack_connection = graph.adj_matrix[cur_node[0]][neighbor]
            for node, node_obj in enumerate(graph.stack_list[neighbor].node_stack):
                edge_weight = stack_connection[cur_node[1]][node]
                node_cost = node_value_selection[weight_variable](node_obj)
                new_distance = cur_dist + node_cost + edge_weight
                heapq.heappush(que, (new_distance, cur_path + ((neighbor, node),)))

    raise ValueError(
        "These operations are not computable with this hardware. Change hardware or \
        algorithms for the hardware. Quick_huristic also may be on, leading to missed paths"
    )


def pathfinding_node_selection(subgraphs, weight_variable):
    """apply roling_dijkstra to each subgraph.

    Args:
        subgraphs (StackedGraph)

    Returns:
        Graph
    """
    flat_subgraphs = []
    for i, subgraph in enumerate(subgraphs):
        selected_nodes = _rolling_dijkstra(subgraph, weight_variable=weight_variable)
        # [ (stack_id, node_idx in node_stack) ,]

        subgraph_nodes_list = []
        for selected_node in selected_nodes:
            subgraph_stack = subgraph.stack_list[selected_node[0]]
            subgraph_stack.node_selection = selected_node[1]
            selected_node = subgraph_stack.node_stack[subgraph_stack.node_selection]

            # for flat subgraph
            subgraph_nodes_list.append(
                sg.Node(
                    selected_node.algorithm,
                    subgraph.get_node_obj(subgraph_stack.stack_id),
                )
            )

        # Selection can be based on any weight_variable, but everything afterwards is time based.
        flat_subgraphs.append(sg.Graph(subgraph_nodes_list, weight_variable="time"))

    return flat_subgraphs


# endregion


# region Schedule


def _hardware_synchronize(available_hardware):
    """brings all hardware times up to the max"""
    max_value = max(
        max(inner_dict.values()) for inner_dict in available_hardware.values()
    )

    for sub_dict in available_hardware.values():
        for key in sub_dict:
            sub_dict[key] = max_value


def _scheduling_dijkstra(graph, available_hardware):
    """
    subgraph with mock start node.
    available_hardware initilized to 0
    """
    visited = {idx for idx, val in enumerate(graph.node_list) if not val.parents}
    end_times = {val.stack_id: 0 for val in graph.node_list if not val.parents}
    indegree = {idx: len(stack.parents) for idx, stack in enumerate(graph.node_list)}
    que = []
    for stack_id in graph.in_nodes:
        que.append((graph.id_to_idx[stack_id],))

    while que:
        # select the one that can be done the soonest, parents with the earlies end time
        small_val = np.inf
        small_idx = np.inf
        for idx, v in enumerate(que):
            if end_times[graph.node_list[v[-1]].stack_id] < small_val:
                small_val = end_times[graph.node_list[v[-1]].stack_id]
                small_idx = idx

        cur_path = que.pop(small_idx)
        cur_node = cur_path[-1]

        neighbor_stacks = graph.get_stack_neighbors(cur_node)
        hardware_type = None
        max_parent_end = None

        for neighbor in neighbor_stacks:
            indegree[neighbor] -= 1
            if neighbor not in visited and indegree[neighbor] == 0:
                neighbor_node = graph.node_list[neighbor]

                hardware_type = hw.Hardware.algs[neighbor_node.algorithm].hardware

                parent_end = [end_times[parent] for parent in neighbor_node.parents]
                max_parent_end = max(parent_end)

                # select hardware to use
                less_than = [
                    available
                    for available in available_hardware[hardware_type]
                    if available_hardware[hardware_type][available] <= max_parent_end
                ]

                # no hardware
                if not less_than:
                    selected_hardware = min(
                        available_hardware[hardware_type],
                        key=lambda k: available_hardware[hardware_type][k],
                    )
                else:
                    # select minimum distance away for tightest packing
                    selected_hardware = min(
                        less_than,
                        key=lambda k: max_parent_end
                        - available_hardware[hardware_type][k],
                    )
                    # bring hardware behind hardware up to current
                    available_hardware[hardware_type][
                        selected_hardware
                    ] = max_parent_end

                neighbor_node.hardware_selection = selected_hardware
                assert neighbor_node.start_time is None  # not already scheduled
                neighbor_node.start_time = available_hardware[hardware_type][
                    selected_hardware
                ]

                # add time
                edge_weight = graph.adj_matrix[cur_node][neighbor]
                available_hardware[hardware_type][selected_hardware] += (
                    neighbor_node.time_cost + edge_weight
                )
                new_time = available_hardware[hardware_type][selected_hardware]
                visited.add(neighbor)
                end_times[neighbor_node.stack_id] = new_time
                que.append(cur_path + (neighbor,))


def _time_shift(available_hardware, graph, time):
    """Shift all nodes =

    Args:
        graph (Graph):
        time (int): + for forwared, - for back
    """
    for node in graph.node_list:
        node.start_time += time
        assert node.start_time >= 0, "start times not all positive"

    for sub_dict in available_hardware.values():
        for key in sub_dict:
            sub_dict[key] += time


def _add_residual(original_graph, node_list):
    for node in node_list:
        if (
            node.stack_id in original_graph.residual
        ):  # is ending node to residual connection
            original_stack = original_graph.get_node_obj(node.stack_id)
            node.parents.update(original_stack.parents)


def _add_in_out(original_graph, node_list):
    """add i/o nodes to graph

    Args:
        original_graph (StackGraph): graph from Json
        node_list (list): list of nodes for new graph, scheduled
    """
    all_nodes = set()

    # input_nodes
    for node in node_list:
        all_nodes.add(node.stack_id)
        if node.algorithm != "dot_prod_phu":
            if (
                node.stack_id in original_graph.id_to_idx
                or node.stack_id + 0.1 in original_graph.id_to_idx
            ):
                node.parents = set(node.parents)
                current_parents = {int(parent) for parent in node.parents}
                parents_added = (
                    original_graph.get_node_obj(round(node.stack_id)).parents
                    - current_parents
                )
                node.parents.update(parents_added)

    for in_node in original_graph.in_nodes:
        new_node = original_graph.get_node_obj(in_node).node_stack[0]
        node_list.append(new_node)

    # output_nodes
    for out_node in original_graph.out_nodes:
        out_nod_obj = original_graph.get_node_obj(out_node)
        new_parents = []
        for parent in out_nod_obj.parents:
            if parent in all_nodes:
                new_parents.append(parent)
            else:
                new_parents.append(parent + 0.1)

        new_node = out_nod_obj.node_stack[0]
        new_node.parents = new_parents
        node_list.append(new_node)

    validate.node_list_complete(node_list)


def _schedule_in_out(graph, available_hardware):
    """Schedule the i/o nodes in a graph

    Args:
        graph (Graph)
    """
    min_start_time = np.inf
    for node in graph.in_nodes:
        node_obj = graph.get_node_obj(node)
        childrn = [
            (idx, transfer_cost)
            for idx, transfer_cost in enumerate(graph.adj_matrix[graph.id_to_idx[node]])
            if transfer_cost is not None
        ]
        # graph is not sorted. find earlies start in childrn
        min_node_end_time = np.inf
        min_child_obj = None
        for idx, transfer_cost in childrn:
            child_obj = graph.node_list[idx]
            if (child_obj.start_time - transfer_cost) < min_node_end_time:
                min_node_end_time = child_obj.start_time - transfer_cost
                min_child_obj = child_obj

        node_obj.start_time = min_node_end_time - node_obj.time_cost
        min_start_time = min(min_start_time, node_obj.start_time)
        node_obj.hardware_selection = "memory"

    # schedule out_nodes
    for node in graph.out_nodes:
        node_obj = graph.get_node_obj(node)
        parents = node_obj.parents
        largest = 0
        for parent in parents:
            parent_obj = graph.get_node_obj(parent)
            largest = max(largest, parent_obj.start_time + parent_obj.time_cost)
        node_obj.start_time = largest
        node_obj.hardware_selection = "memory"

    # 0 time pass
    if min_start_time < 0:
        _time_shift(available_hardware, graph, -min_start_time)


def schdeule_nodes(original_graph, subgraphs, available_hardware):
    """
    merges subgraphs
    schedules in and out nodes
    graph = StackedGraph object, original graph
    subgraphs = Subgraph of
    """
    # Schedule subgraphs and merge

    break_points = []
    nodes_seen = set()
    full_node_list = []
    for subgraph in subgraphs:
        _scheduling_dijkstra(subgraph, available_hardware)

        for node in subgraph.node_list:
            if node.algorithm != "start" and node.stack_id not in nodes_seen:
                if 0 in node.parents:
                    node.parents.remove(0)
                full_node_list.append(node)
                nodes_seen.add(node.stack_id)

        _hardware_synchronize(available_hardware)
        break_points.append(
            max(max(inner_dict.values()) for inner_dict in available_hardware.values())
        )

    _add_residual(original_graph, full_node_list)
    validate.merge_i_o(full_node_list, original_graph)
    _add_in_out(original_graph, full_node_list)

    graph = sg.Graph(full_node_list, weight_variable="time")
    _schedule_in_out(graph, available_hardware)

    for node in graph.node_list:
        if node.stack_id not in graph.in_nodes and node.stack_id not in graph.out_nodes:
            assert node.hardware_selection is not None
            assert node.start_time is not None

    end_time = round(
        max(max(inner_dict.values()) for inner_dict in available_hardware.values()),
        5,
    )

    return (
        graph,
        end_time,
        break_points,
    )


# endregion


# region Expansion


def _group_dot_products(m1, m2):
    """given to tensors, returns dotproducts grouped by most common vector used

    Args:
        m1 (tuple): m1 shape
        m2 (tuple): m2 shape

    Returns:
        dict: common_opperand: [unique_operands]
    """
    groups = {}
    if m1[-2] <= m2[-2]:  # a <= c in axb @ bxc
        for dot_prod in pa.nd_tensor_to_dot(m1, m2):
            groups.setdefault(dot_prod[0], (dot_prod[2], []))[1].append(dot_prod[1])
    else:  # a > c
        for dot_prod in pa.nd_tensor_to_dot(m1, m2):
            groups.setdefault(dot_prod[1], (dot_prod[2], []))[1].append(dot_prod[0])
    return groups


def _matmul_graph(node):
    """given a photonic node, create expanded computational graph to replace it

    Args:
        node (Node): Photonic algorithm
    """
    m1, m2 = node.input_shapes
    # dot_prod_groups = _group_dot_products(m1, m2)
    mtrx_mtrx = list(pa.nd_tensor_to_matx(tuple(m1), tuple(m2)))

    split_node = sg.Node("split", node.stack)
    split_node.stack_id -= 0.1
    split_node.output_shapes = []

    merge_node = sg.Node("split", node.stack)
    merge_node.stack_id += 0.1
    merge_node.parents = {}
    merge_node.input_shapes = []

    node_expansion_func = pa.node_expansion[node.algorithm]

    subnodes = []
    for mtrx_product in mtrx_mtrx:
        subnodes.append(node_expansion_func(node, *mtrx_product))

    for subnode in subnodes:
        split_node.output_shapes += subnode.input_shapes
        merge_node.input_shapes += subnode.output_shapes

    merge_node.parents = {subnode.stack_id for subnode in subnodes}
    return [split_node, merge_node] + subnodes


def _update_children(graph, node_idx):
    """propogate patrent id change to children

    Args:
        graph (Graph):
        node_idx (int): location of node that changed
    """
    node_obj = graph.node_list[node_idx]
    for child_idx in graph.get_stack_neighbors(node_idx):
        child_obj = graph.node_list[child_idx]
        child_obj.parents = [
            parent + 0.1 if parent == node_obj.stack_id else parent
            for parent in child_obj.parents
        ]


def expand_nodes(flat_subgraphs):
    """given a flat_graph, replace all photonic nodes with their complete subgraphs

    Args:
        flat_graph (Graph): entire Computation
        flat_subgraphs (Graph):
    """
    new_subgraphs = []
    for subgraph in flat_subgraphs:
        new_subgraph_node_list = []

        # add replacement nodes
        for node_idx, node in enumerate(subgraph.node_list):
            if node.algorithm in pa.node_expansion:
                replacement_nodes = _matmul_graph(node)
                new_subgraph_node_list += replacement_nodes
                _update_children(subgraph, node_idx)

        # add rest, some have been modified
        for node_idx, node in enumerate(subgraph.node_list):
            if node.algorithm not in pa.node_expansion:
                new_subgraph_node_list.append(node)

        new_subgraphs.append(sg.Graph(new_subgraph_node_list, subgraph.weight_variable))

    return new_subgraphs


# endregion


# regon Thresholding


def _get_all_in_connection_cost(stacked_graph, moc_stack):
    dependancys = [(inp, moc_stack.stack_id) for inp in moc_stack.parents]

    all_parent_connections = list(
        stacked_graph.make_connection(*dep) for dep in dependancys
    )

    # [ [ [ [ 3.33333333e-10, 1.03333333e-08 ] ] ], [[[3.33333333e-10, 1.03333333e-08]]] ]
    #        |end stack idx-|
    #     |--------parrent stack idx-----------|
    #   |--------------parrent stack-------------|

    totals_per_node = [0] * len(moc_stack)
    for parrent in all_parent_connections:
        parrent_node = parrent[0]  # parent stacks only have 1 node
        for end_node_idx, edge in enumerate(parrent_node):
            totals_per_node[end_node_idx] += edge
    return totals_per_node


def _get_all_out_connection_cost(stacked_graph, moc_stack):
    dependancys = [
        (moc_stack.stack_id, inp)
        for inp in stacked_graph.get_stack_neighbors(moc_stack.stack_id)
    ]

    all_child_connections = list(
        stacked_graph.make_connection(*dep) for dep in dependancys
    )

    # [ [ [ [ 3.33333333e-10 ], [ 1.03333333e-08 ] ] ] ]
    #        |end stack idx|
    #       |-cur stack idx--|
    #   |------------------child stack---------------| only one child

    totals_per_node = [0] * len(moc_stack)
    for child in all_child_connections:
        for cur_node_idx, current_node in enumerate(child):
            totals_per_node[cur_node_idx] += current_node[
                0
            ]  # child stacks only have 1 node

    return totals_per_node


def _get_stack_threshold(
    stacked_graph,
    stack,
    weight_variable="time",
    plot_len_cost=False,
    plot_arithmatic_intensity=False,
):
    sequence_len = []
    algs = {}

    arithmatic_intensity = []
    electronic = []
    photonic = []

    threshold = np.inf
    for moc_sequence_len in range(4096):  # 4096 for llama
        moc_stack = sg.Stack(
            stack.stack_id,
            stack.parents,
            sg.get_moc_size(stack.input_shapes, 6, moc_sequence_len),
            sg.get_moc_size(stack.output_shapes, 6, moc_sequence_len),
            stack.tvm_func,
            relay_node=stack.relay_node,
        )
        min_cost = np.inf
        min_cost_alg = None
        total_connection_in_per_node = _get_all_in_connection_cost(
            stacked_graph, moc_stack
        )
        total_connection_out_per_node = _get_all_out_connection_cost(
            stacked_graph, moc_stack
        )
        for idx, node in enumerate(moc_stack):
            total_cost = (
                node_value_selection[weight_variable](node)
                + total_connection_in_per_node[idx]
                + total_connection_out_per_node[idx]
            )
            if total_cost < min_cost:
                min_cost = total_cost
                min_cost_alg = node.algorithm

            algs.setdefault(node.algorithm, []).append(total_cost)
            if "phu" in node.algorithm:
                photonic.append(total_cost)
            else:
                electronic.append(total_cost)

        sequence_len.append(moc_sequence_len)
        intensity = hw.arithmatic_intensity_matmul(
            moc_stack.input_shapes, moc_stack.output_shapes
        )
        arithmatic_intensity.append(intensity)

        if "phu" in min_cost_alg:
            threshold = moc_sequence_len
            if not plot_len_cost and not plot_arithmatic_intensity:
                break

    if plot_len_cost:
        dc.plot_len_cost(sequence_len, algs, weight_variable)

    if plot_arithmatic_intensity:
        dc.plot_arithmatic_intensity(
            arithmatic_intensity, [p - e for p, e in zip(photonic, electronic)]
        )

    return threshold


def threshold_nodes(stacked_graph, weight_variable="time"):
    count = 0
    threshold_values = {}
    for stack in stacked_graph:
        if len(stack) == 1:
            threshold_values[stack.stack_id] = None
        else:
            threshold_sequence_len = _get_stack_threshold(
                stacked_graph,
                stack,
                weight_variable=weight_variable,
                plot_len_cost=False,
                plot_arithmatic_intensity=False,
            )
            threshold_values[stack.stack_id] = threshold_sequence_len
    return threshold_values


# endregion
