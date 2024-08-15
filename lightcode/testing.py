"""
Testing and validation
"""

def group_validate(graph, groups):
    """ensures every node parents are included in the group.
    exception to load and store nodes, which can have odd dependancies
    """
    for i, lst in enumerate(groups):
        load_instructions = {
            stack.stack_id for stack in graph.stack_list if stack.opp == "memory"
        }
        included = set(lst)
        for stack in lst[1:]:
            for parent in graph.stack_list[stack].parents:
                if parent in load_instructions:
                    continue
                # assert parent in included
                if parent not in included:
                    if graph.stack_list[parent].opp and graph.stack_list[stack].opp in ['slice', 'add']:
                        graph.stack_list[stack].residual=True
                    else:
                        assert False
    return True


def node_list_complete(node_list):
    """all node parents are present in the list

    Args:
        node_list (list): list of Node Objects

    Returns:
        bool:
    """
    all_nodes = set()
    for node in node_list:
        assert (
            node.stack_id not in all_nodes
        ), f"list has repetitious stack node: {node.stack_id}"
        all_nodes.add(node.stack_id)

    for node in node_list:
        assert all(
            parent in all_nodes for parent in node.parents
        ), "not all parents are in the list"
    return True


def merge_i_o(full_node_list, original_graph):
    """Disreguarding i/o, do subgraph and parent graph node parents aggree

    Args:
        full_node_list (list): list of Node Objects
        original_graph (Grph): Graph made from Relay IR

    Returns:
        bool:
    """
    total = 0
    for node in full_node_list:
        if node.algorithm != "dot_prod_phu":
            if (
                node.stack_id in original_graph.id_to_idx
                or node.stack_id + 0.1 in original_graph.id_to_idx
            ):
                total += 1
                original_node = original_graph.get_node_obj(round(node.stack_id))

                original_parents = (
                    original_node.parents
                    - original_graph.in_nodes
                    - original_graph.out_nodes
                )
                this_parent = {int(parent) for parent in node.parents}
                assert this_parent == original_parents
    return True


def schedule_validate(schedule_df):
    # check for overlaps
    stagnent_time = {}
    unique_hardware = schedule_df["hardware"].unique()
    unique_hardware = unique_hardware[unique_hardware != "memory"]

    for hardware in unique_hardware:
        filter = schedule_df["hardware"] == hardware
        hw_filtered = schedule_df.loc[filter]
        hw_sorted = hw_filtered.sort_values(by="start")

        sparse = 0
        last_end = 0
        for index, row in hw_sorted.iterrows():
            start = row["start"]
            assert start >= last_end
            sparse += start - last_end

            last_end = row["end"]

        stagnent_time[hardware] = sparse

    # print("... No Overlaps ...")
    return stagnent_time
