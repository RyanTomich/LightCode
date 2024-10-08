import input_validation as validate
import re
import math

# Code format
header = f"|start_time |---|hardware|---|core|---|________________function________________|---|_________opp________|---|_io___\n\n"
template_base = "{start_time:.10f}     {hardware:<8}     {core:4d}     {func:<40}     {opp:<20}     "
template_out = "o{out_var:<5} "
template_in = "i{in_var:<5} "


def get_class_obj(hardware_core):
    if "<" not in hardware_core:
        return hardware_core, 0

    match = re.search(r">(\d+)$", hardware_core)
    if match:
        core = int(match.group(1))
    else:
        raise ValueError("Invalid object string format")

    match = re.match(r"<hardware.(.*) object at", hardware_core)
    if match:
        class_name = match.group(1)
    else:
        raise ValueError("Invalid object string format")

    return class_name, core


def code_gen(scheduled_flat_graph):

    def dump_variables(node):
        for parent in node.parents:
            parent_node_id = scheduled_flat_graph.get_node_obj(parent).node_id
            node_children[parent_node_id].remove(node.node_id)
            if len(node_children[parent_node_id]) == 0:
                variable_que.extend(node_variables[parent_node_id])

    def get_variable():
        if not variable_que:
            var = variable_counter[0]
            variable_counter[0] += 1
            return var
        else:
            return variable_que.pop(0)

    variable_que = []
    variable_counter = [0]

    sorted_nodes = scheduled_flat_graph.get_sorted_nodes()

    # assumes parrents are earlier in sorted nodes than children
    seen = set()
    node_children = {}
    for node in sorted_nodes:
        node_children.setdefault(node.node_id, [])
        for parent in node.parents:
            node_children[scheduled_flat_graph.get_node_obj(parent).node_id].append(
                node.node_id
            )

    with open("code.txt", "w") as file:
        node_variables = {}
        file.write(header)
        for node in sorted_nodes:

            dump_variables(node)

            input_variables = []
            for parent in node.parents:
                parent_obj = scheduled_flat_graph.get_node_obj(parent)
                parent_node_id = parent_obj.node_id
                assert parent_node_id in seen

                fractional_part, _ = math.modf(parent_obj.stack_id)
                if math.isclose(fractional_part, 0.9, abs_tol=1e-9):
                    input_variables.extend(node_variables[parent_node_id][:2])
                    node_variables[parent_node_id].pop(0)
                    node_variables[parent_node_id].pop(0)
                else:
                    input_variables.extend(node_variables[parent_node_id])
            # print(len(node.output_shapes))
            output_variables = [get_variable() for _ in range(len(node.output_shapes))]

            node_variables[node.node_id] = output_variables

            hardware, core = get_class_obj(node.hardware_selection)

            seen.add(node.node_id)

            if hardware == "memory":
                function = "memory"
            elif hardware == "PHU":
                function = node.algorithm
            elif node.stack.tvm_func is not None:
                function = node.stack.tvm_func

            file.write(
                template_base.format(
                    start_time=node.start_time,
                    hardware=hardware,
                    core=core,
                    func=function,
                    opp=node.algorithm,
                )
            )

            for var in output_variables:
                file.write(template_out.format(out_var=var))
            for var in input_variables:
                file.write(template_in.format(in_var=var))

            file.write("\n")

    schedule_df = scheduled_flat_graph.create_schedule_data(write=True)
    validate.schedule_validate(schedule_df)
