"""
Functions for expanding photnic matrix multiplication
"""

from lightcode import hardware
from lightcode import stack_graph


# Create Dot Products
def _matmul(m1, m2, preamble):
    """createss dot products from 2 matricies

    Args:
        m1 (tuple): shape of m1
        m2 (tuple): shape of m2
        preamble (list): list of indexing path to current matrix.

    Yields:
        srt: dot product with indexing
    """
    for row in range(m1[-2]):
        for col in range(m2[-2]):
            yield (
                (1,) + preamble + (row, ":"),
                (2,) + preamble + (col, ":"),
                m1[-1],
            )  # (m1.v1, m2.v2, size)
            # yield f'm1{preamble + [row, ':']} * m2{preamble + [col, ':']}  -  {m1[-1]}'


def nd_tensor_to_dot(m1, m2, preamble=()):
    """Breaks tensor down to the level of matrix multiplication

    Args:
        m1 (tuple): shape of m1
        m2 (tuple): shape of m2
        preamble (list, optional): list of indexing path to current matrix. Defaults to [].

    Yields:
        srt: dot product with indexing
    """
    if len(m1) == 2:
        yield from _matmul(m1, m2, preamble)
    else:
        for dimention in range(m1[0]):
            preamble = preamble + (dimention,)
            yield from nd_tensor_to_dot(m1[1:], m2[1:], preamble=preamble)
            preamble = preamble[:-1]


def nd_tensor_to_matx(m1, m2, preamble=()):
    """Breaks tensor product down to matrix products"""
    if len(m1) == 2:
        yield (
            preamble,  # index
            m1,  # size m1
            m2,  # size m2
        )
    else:
        for dimention in range(m1[0]):
            preamble = preamble + (dimention,)
            yield from nd_tensor_to_matx(m1[1:], m2[1:], preamble=preamble)
            preamble = preamble[:-1]


# Expansion Functinos


def _multiplex_groups(local_hardware, size, common_operand, unique_operands):
    """group nodes into max multiplex

    Args:
        size (int): length of dot product
        common_operand (list):
        unique_operands (list of lists):

    Yields:
        tupele: (size, common, list(unique))
    """
    full = int(len(unique_operands) / local_hardware.num_numtiplex)
    start = 0
    end = local_hardware.num_numtiplex
    for _ in range(full):
        yield (size, common_operand, unique_operands[start:end])
        start = end
        end += local_hardware.num_numtiplex

    assert end >= len(unique_operands)
    if len(unique_operands) % local_hardware.num_numtiplex:
        assert end > len(unique_operands)
        yield (size, common_operand, unique_operands[start:])


def _task_para_node_gen(node, index, m1, m2):
    """takes group of matrix matrix products and creates list of nodes for a computational graph

    Args:
        size (int): lenth of each dorproduct
        common_operand (tup): m1()
        unique_operands (list): list of m2()
    """
    assert len(m1) == len(m2) == 2  # two dimentions

    local_hardware = hardware.Hardware.algs[node.algorithm].alg_hardware
    subnode = stack_graph.Node("matrix_matrix_phu", node.stack)
    subnode.parents = [node.stack_id - 0.1]
    subnode.index = index
    subnode.input_shapes = [list(m1), list(m2)]
    subnode.output_shapes = [[m1[0], m2[0]]]
    hardware.NODE_COUNT += 1
    subnode.stack_id = hardware.NODE_COUNT

    subnode.time_cost = hardware.Hardware.algs[subnode.algorithm].time_cost(
        subnode.input_shapes, subnode.output_shapes
    )
    subnode.energy_cost = hardware.Hardware.algs[subnode.algorithm].energy_cost(
        subnode.input_shapes, subnode.output_shapes
    )

    return subnode

    # for multiplex in _multiplex_groups(local_hardware, size, common_operand, unique_operands):
    #     size, common_operand, unique_subset = multiplex

    #     subnode = stacked_graph.Node("dot_prod_phu", node.stack)
    #     subnode.parents = [node.stack_id - 0.1]

    #     subnode.input_shapes = [
    #         [len(unique_subset) + 1, size]
    #     ]  # multiplex vectors + common vector
    #     subnode.output_shapes = [[1, len(unique_subset) + 1]]

    #     hardware.NODE_COUNT += 1
    #     subnode.stack_id = hardware.NODE_COUNT
    #     subnodes.append(subnode)

    # return subnodes


def _dynamic_para_node_gen(common_operand, unique_operands):
    pass


node_expansion = {  # node algorithm to it's expansion function
    "task_para_matmul_phu": _task_para_node_gen,
    "task_para_dense_phu": _task_para_node_gen,
    "task_para_pack_phu": _task_para_node_gen,
    "dynamic_para_matmul_phu": _dynamic_para_node_gen,
    "dynamic_para_dense_phu": _dynamic_para_node_gen,
    "dynamic_para_pack_phu": _dynamic_para_node_gen,
}
