"""
simulations constants and definitions
"""

import math
import numpy as np

DEBUG_PRINT = False


# Scheduling
def creat_available_hardware(hardware_dict):
    """Given the machiene hardware, create hardware dict

    Args:
        hardware_dict (dict): hardware type to number of that hardware

    Returns:
        dict: each individual hardware to time (0)
    """
    hardware = {}
    for hw, count in hardware_dict.items():
        in_dict = {}
        for i in range(count):
            in_dict[f"{hw}{i}"] = 0
        hardware[hw] = in_dict
    return hardware


def initilize_hardware():
    """Creates global definition of the hardware type to count.
    This is here to explicitly create hardware once when needed
    """
    global available_hardware
    available_hardware = creat_available_hardware(
        {"CPU": CPU_CORES, "PHU": PHU_CORES, "SRAM": 1}
    )


# Time functions
def ten_elm(tensor_shape):
    """
    Args:
        tensor_shape (lsit or tuple)

    Returns:
        int: number of elements
    """
    ans = 1
    for dimention in tensor_shape:
        ans *= dimention
    return ans


def all_elm(i, o):
    return {"CPU": ten_elm(o[0])}


def constnat(c):
    return lambda i, o: {"CPU": c}


def elm_const(matrix, const=1):
    return ten_elm(matrix) * const


## CPU
def cpu_matmul_cycles(i, o):
    num_dot_products = ten_elm(o[0])
    length_dot_products = i[0][-1]
    ops_per_mac = 2  # multiplicaiton and addition
    return {"CPU": num_dot_products * length_dot_products * ops_per_mac}


def cpu_matmul_energy(i, o):
    num_dot_products = ten_elm(o[0])
    length_dot_products = i[0][-1]
    return num_dot_products * length_dot_products * GPU_MAC


## PHU
def phu_matmul_task_para_cycles(i, o):
    num_dot_products = ten_elm(o[0])
    length_dot_products = i[0][-1]
    phu_cycles = (
        math.ceil(math.ceil(num_dot_products / PHU_MULTIPLEX) / PHU_CORES)
        * length_dot_products
    )
    return {"PHU": phu_cycles}


def phu_matmul_task_para_energy(i, o):
    num_dot_products = ten_elm(o[0])
    length_dot_products = i[0][-1]
    return num_dot_products * length_dot_products * PHU_MAC


def phu_matmul_dynamic_para_time(i, o):  # TODO
    return {"PHU": np.inf}


def func():
    print("placeholder")


def get_edge_val(graph, start_node, end_node, weight_variable):
    """Calculates cost between hardware. Accounts for trips to and from SRAM

    Args:
        graph (Graph):
        start_node (Node): ending node of directed edge
        end_node (Node): ending node of directed edge
        weight_variable (str): time or energy

    Returns:
        int: total cost in seconds or jules (depending on weight_variable)
    """

    num_transfer, bit_transfer = graph._bit_transfer(start_node)

    start_hw = start_node.get_algo_info("hardware")
    end_hw = end_node.get_algo_info("hardware")
    hw_connection = tuple((start_hw, end_hw))

    # one way connection
    if any(i in ["start", "SRAM"] for i in hw_connection):
        return hw_intercon[hw_connection].get_transfer_cost(
            weight_variable, num_transfer, bit_transfer
        )

    # must go through SRAM
    else:
        to_sram = hw_intercon[(hw_connection[0], "SRAM")].get_transfer_cost(
            weight_variable, num_transfer, bit_transfer
        )
        from_sram = hw_intercon[("SRAM", hw_connection[1])].get_transfer_cost(
            weight_variable,
            num_transfer,
            bit_transfer,
        )

        return to_sram + from_sram


# region Constants
NODE_COUNT = 0

# cores
PHU_CORES = 64
PHU_MULTIPLEX = 20
CPU_CORES = 1

# Time
CPU_CLOCK_SPEED = 10**8  # .1Ghz
PHU_CLOCK_SPEED = 10**10  # 10 Ghz

CPU_CLOCK_PERIOD_SECONDS = 1 / CPU_CLOCK_SPEED
PHU_CLOCK_PERIOD_SECONDS = 1 / PHU_CLOCK_SPEED

MEMORY_TRANSFER_WIDTH = 32  # bits per cycle
DAC_ADC_DELAY = 1 * CPU_CLOCK_PERIOD_SECONDS  # 10 nano-seconds

BITS_PER_NUM = 32

# Power
PICO_JOULE = 10**-12

JOULE_PER_CYCLE = 1 * PICO_JOULE

DRAM_READ = 160 * PICO_JOULE
DRAM_WRITE = 160 * PICO_JOULE
HBM_READ = 40 * PICO_JOULE
HBM_WRITE = 40 * PICO_JOULE
SRAM_READ = 12 * PICO_JOULE
SRAM_WRITE = 12 * PICO_JOULE
LOCAL_READ = 1 * PICO_JOULE
LOCAL_WRITE = 1 * PICO_JOULE
GPU_MAC = 0.1 * PICO_JOULE
PHU_MAC = 0.04 * PICO_JOULE

DAC_POWER = 3.18 * PICO_JOULE
ADC_POWER = 1.6 * PICO_JOULE

# endregion


# region node selection

cycle_to_time_funcs = {
    "CPU": lambda x: x * CPU_CLOCK_PERIOD_SECONDS,
    "PHU": lambda x: x * PHU_CLOCK_PERIOD_SECONDS,
    "HBM": lambda x: x * CPU_CLOCK_PERIOD_SECONDS,
}

node_value_selection = {
    "time": lambda node: node.time_cost,
    "energy": lambda node: node.energy_cost,
}

# data_pipeline_algorithms = {
#     "matmul"
#     "dense"
#     "pack"
#     "task_para_matmul_phu"
#     "task_para_dense_phu"
#     "task_para_pack_phu"
#     "dynamic_para_matmul_phu"
#     "dynamic_para_dense_phu"
#     "dynamic_para_pack_phu"
# }


class HardwareAlgorithm:
    def __init__(
        self,
        opp,
        hardware,
        function,
        cycle_function,
        energy_function=lambda i, o: JOULE_PER_CYCLE,
    ):
        self.opp = opp
        self.hardware = hardware
        self.func = function
        self.cycle_function = cycle_function

        # default value untill energy for all other algs
        self.energy_function = energy_function

    def cycle_to_s(self, cost: dict) -> int:
        total = 0
        for hardware, cycles in cost.items():
            total += cycle_to_time_funcs[hardware](cycles)
        return total

    def time_cost(self, i, o):
        return self.cycle_to_s(self.cycle_function(i, o))

    def energy_cost(self, i, o):
        return self.energy_function(i, o)


hardware_algs = {
    "add": HardwareAlgorithm("add", "CPU", func, all_elm),
    "subtract": HardwareAlgorithm("subtract", "CPU", func, all_elm),
    "multiply": HardwareAlgorithm("multiply", "CPU", func, all_elm),
    "divide": HardwareAlgorithm("divide", "CPU", func, all_elm),
    "sqrt": HardwareAlgorithm("sqrt", "CPU", func, all_elm),
    "rsqrt": HardwareAlgorithm("rsqrt", "CPU", func, all_elm),
    "relu": HardwareAlgorithm("relu", "CPU", func, all_elm),
    "tanh": HardwareAlgorithm(
        "tanh", "CPU", func, lambda i, o: {"CPU": ten_elm(o[0]) * 4}
    ),
    "power": HardwareAlgorithm("power", "CPU", func, all_elm),
    "transpose": HardwareAlgorithm("transpose", "CPU", func, all_elm),
    "nop": HardwareAlgorithm("nop", "CPU", func, constnat(1)),
    "less": HardwareAlgorithm("less", "CPU", func, constnat(1)),
    "take": HardwareAlgorithm("take", "CPU", func, constnat(1)),
    "split": HardwareAlgorithm("split", "SRAM", func, constnat(1)),
    "mean": HardwareAlgorithm(
        "mean", "CPU", func, lambda i, o: {"CPU": (i[0][-1] + 1) * i[0][-2]}
    ),
    "softmax": HardwareAlgorithm(
        "softmax", "CPU", func, lambda i, o: {"CPU": ten_elm(o[0]) * 6}
    ),
    "matmul": HardwareAlgorithm(
        "matmul", "CPU", func, cpu_matmul_cycles, cpu_matmul_energy
    ),
    "dense": HardwareAlgorithm(
        "dense", "CPU", func, cpu_matmul_cycles, cpu_matmul_energy
    ),
    "pack": HardwareAlgorithm(
        "pack", "CPU", func, cpu_matmul_cycles, cpu_matmul_energy
    ),
    "where": HardwareAlgorithm("where", "CPU", func, constnat(1)),
    "erf": HardwareAlgorithm("erf", "CPU", func, constnat(1)),
    "task_para_matmul_phu": HardwareAlgorithm(
        "matmul", "PHU", func, phu_matmul_task_para_cycles, phu_matmul_task_para_energy
    ),
    "task_para_dense_phu": HardwareAlgorithm(
        "dense", "PHU", func, phu_matmul_task_para_cycles, phu_matmul_task_para_energy
    ),
    "task_para_pack_phu": HardwareAlgorithm(
        "pack", "PHU", func, phu_matmul_task_para_cycles, phu_matmul_task_para_energy
    ),
    # "dynamic_para_matmul_phu": HardwareAlgorithm(  # TODO
    #     "matmul", "PHU", func, phu_matmul_dynamic_para_time, lambda i, o: np.inf
    # ),
    # "dynamic_para_dense_phu": HardwareAlgorithm(  # TODO
    #     "dense", "PHU", func, phu_matmul_dynamic_para_time, lambda i, o: np.inf
    # ),
    # "dynamic_para_pack_phu": HardwareAlgorithm(  # TODO
    #     "pack", "PHU", func, phu_matmul_dynamic_para_time, lambda i, o: np.inf
    # ),
    "get_dram": HardwareAlgorithm(
        "null",
        "HBM",
        func,
        lambda i, o: {
            # "HBM": sum(ten_elm(a) for a in o) * BITS_PER_NUM / MEMORY_TRANSFER_WIDTH
            "HBM": 0  # 0 if assuming model is preloaded to HMB
        },
        lambda i, o: sum(ten_elm(a) for a in o) * DRAM_READ
        + sum(ten_elm(a) for a in o) * HBM_WRITE,
    ),
    "start": HardwareAlgorithm("start", "start", func, constnat(1)),
    "dot_prod_phu": HardwareAlgorithm(
        "dot_prod",
        "PHU",
        func,
        lambda i, o: {"PHU": i[0][-1]},
        lambda i, o: i[0][-1] * PHU_MAC,
    ),
}

# endregion


# region Edges


class HardwareConnection:
    def __init__(self, time_cost_per_transfer, energy_cost_per_number):
        """create functions for cost

        Args:
            time_cost_func (int): time cost in seconds per bit
            energy_cost_func (int): energy cost in joules per bit
        """
        self.time_cost_func = lambda n, b: time_cost_per_transfer
        self.energy_cost_func = lambda n, b: n * energy_cost_per_number
        self.var_to_func = {
            "time": self.time_cost_func,
            "energy": self.energy_cost_func,
        }

    def get_transfer_cost(self, weight_variable, num_transfer, bit_transfer):
        return self.var_to_func[weight_variable](num_transfer, bit_transfer)


hw_intercon = {
    ("HBM", "SRAM"): HardwareConnection(
        CPU_CLOCK_PERIOD_SECONDS, HBM_READ + SRAM_WRITE
    ),
    ("SRAM", "CPU"): HardwareConnection(
        CPU_CLOCK_PERIOD_SECONDS, SRAM_READ + LOCAL_WRITE + LOCAL_READ
    ),
    ("SRAM", "PHU"): HardwareConnection(
        CPU_CLOCK_PERIOD_SECONDS + DAC_ADC_DELAY,
        SRAM_READ + LOCAL_WRITE + LOCAL_READ + DAC_POWER,
    ),
    ("CPU", "SRAM"): HardwareConnection(
        CPU_CLOCK_PERIOD_SECONDS, LOCAL_WRITE + LOCAL_READ + SRAM_WRITE
    ),
    ("PHU", "SRAM"): HardwareConnection(
        CPU_CLOCK_PERIOD_SECONDS + DAC_ADC_DELAY,
        ADC_POWER + LOCAL_WRITE + LOCAL_READ + SRAM_WRITE,
    ),
    ("SRAM", "HBM"): HardwareConnection(DAC_ADC_DELAY, SRAM_READ + HBM_WRITE),
    # start nodes
    ("start", "CPU"): HardwareConnection(0, 0),
    ("start", "PHU"): HardwareConnection(0, 0),
    ("start", "SRAM"): HardwareConnection(0, 0),
}

# endregion
