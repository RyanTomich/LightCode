import math


# time and dimentions
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


# i, o functions
def all_elm(i, o):
    return ten_elm(o[0])


def all_elm_const(c):
    return lambda i, o: ten_elm(o[0]) * c


def constnat(c):
    return lambda i, o: c


def standard_energy(i, o):
    return JOULE_PER_CYCLE


def energy_per_cycle_func_gen(funcion):
    return lambda i, o: funcion(i, o) * JOULE_PER_CYCLE


def elm_const(matrix, const=1):
    return ten_elm(matrix) * const


def recursive_contains_num(nested_list, func):
    for element in nested_list:
        if isinstance(element, list):
            if recursive_contains_num(element, func):
                return True
        elif func(element) is True:
            return True
    return False


def arithmatic_intensity_matmul(i, o):
    num_dot_products = ten_elm(o[0])
    length_dot_products = i[0][-1]
    total_macs = num_dot_products * length_dot_products

    total_num_in = ten_elm(i[0]) + ten_elm(i[1])
    total_num_out = ten_elm(o[0])

    return total_macs / (total_num_in + total_num_out + 1e-9)


# endregion


# region Hardware
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


def initilize_hardware(hardware):
    """creates intercon functions

    Args:
        hw (list): list of hardware instances of the system
    """

    # MEMORY_CLOCK = 6 * 10**9  # 60**9, 6 Ghz

    sram = SRAM(MEMORY_CLOCK)
    hbm = HBM(MEMORY_CLOCK)
    start = Start(MEMORY_CLOCK)

    available_cores = {hw: hw.num_cores for hw in hardware}
    available_cores[sram] = 1

    available_hardware = creat_available_hardware(available_cores)

    Hardware.intercon = {
        (HBM, SRAM): HardwareConnection(sram.clock_period, HBM_READ + SRAM_WRITE),
        (SRAM, HBM): HardwareConnection(DAC_ADC_DELAY, SRAM_READ + HBM_WRITE),
        # start nodes
        (Start, CPU): HardwareConnection(0, 0),
        (Start, PHU): HardwareConnection(0, 0),
        (Start, SRAM): HardwareConnection(0, 0),
    }
    hw_types = {type(hw_obj) for hw_obj in hardware}
    for hw_obj in hardware:
        if isinstance(hw_obj, CPU):
            Hardware.intercon.update(
                {
                    (SRAM, CPU): HardwareConnection(
                        sram.clock_period, SRAM_READ + LOCAL_WRITE + LOCAL_READ
                    ),
                    (CPU, SRAM): HardwareConnection(
                        sram.clock_period, LOCAL_WRITE + LOCAL_READ + SRAM_WRITE
                    ),
                }
            )
        if isinstance(hw_obj, PHU):
            Hardware.intercon.update(
                {
                    (SRAM, PHU): HardwareConnection(
                        sram.clock_period + DAC_ADC_DELAY,
                        SRAM_READ + LOCAL_WRITE + LOCAL_READ + DAC_POWER,
                    ),
                    (PHU, SRAM): HardwareConnection(
                        sram.clock_period + DAC_ADC_DELAY,
                        ADC_POWER + LOCAL_WRITE + LOCAL_READ + SRAM_WRITE,
                    ),
                }
            )

    return available_hardware


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

    num_transfer, bit_transfer = graph._bit_transfer(start_node, direction="out")

    start_hw = type(start_node.get_algo_info("hardware"))
    end_hw = type(end_node.get_algo_info("hardware"))
    hw_connection = tuple((start_hw, end_hw))

    # one way connection
    if any(i in [Start, SRAM] for i in hw_connection):
        return Hardware.intercon[hw_connection].get_transfer_cost(
            weight_variable, num_transfer, bit_transfer
        )

    # must go through SRAM
    else:
        to_sram = Hardware.intercon[(hw_connection[0], SRAM)].get_transfer_cost(
            weight_variable, num_transfer, bit_transfer
        )
        from_sram = Hardware.intercon[(SRAM, hw_connection[1])].get_transfer_cost(
            weight_variable,
            num_transfer,
            bit_transfer,
        )

        return to_sram + from_sram


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


class HardwareAlgorithm:
    def __init__(self, opp, cost):  # hardware: (time, energy)
        self.opp = opp
        self.cost = cost
        self.hardware = next(
            iter(cost.keys())
        )  # will need to change for multi hardware algorithms

    def time_cost(self, i, o):
        return sum(
            hardware.clock_period * cost_tup[0](i, o)
            for hardware, cost_tup in self.cost.items()
        )

    def energy_cost(self, i, o):
        return sum(cost_tup[1](i, o) for hardware, cost_tup in self.cost.items())


class Hardware:
    _universal_hardware_ID = 0
    algs = {}
    intercon = {}

    def __init__(self, clock_speed):
        self.clock_speed = clock_speed
        self.clock_period = 1 / clock_speed
        Hardware._universal_hardware_ID += 1
        self._initialize_algs()

    def __hash__(self):
        return hash(Hardware._universal_hardware_ID)

    def _initialize_algs(self):
        Hardware.algs.update(self.algs)

    def _hardware_reset():
        Hardware._universal_hardware_ID = 0
        Hardware.algs = {}
        Hardware.intercon = {}


class PHU(Hardware):
    def __init__(self, clock_speed, num_cores, num_multiplex):
        self.num_numtiplex = num_multiplex
        self.num_cores = num_cores
        self.mac_energy = PHU_MAC
        self.algs = {
            "task_para_matmul_phu": HardwareAlgorithm(
                "matmul",
                {
                    self: (
                        self._phu_matmul_task_para_cycles,
                        self._phu_matmul_task_para_energy,
                    )
                },
            ),
            "task_para_dense_phu": HardwareAlgorithm(
                "dense",
                {
                    self: (
                        self._phu_matmul_task_para_cycles,
                        self._phu_matmul_task_para_energy,
                    )
                },
            ),
            "task_para_pack_phu": HardwareAlgorithm(
                "pack",
                {
                    self: (
                        self._phu_matmul_task_para_cycles,
                        self._phu_matmul_task_para_energy,
                    )
                },
            ),
            "matrix_matrix_phu": HardwareAlgorithm(
                "matrix_prod",
                {
                    self: (
                        self._phu_matmul_task_para_cycles,
                        self._phu_matmul_task_para_energy,
                    )
                },
            ),
            "dot_prod_phu": HardwareAlgorithm(
                "dot_prod",
                {
                    self: (
                        lambda i, o: i[0][-1],
                        lambda i, o: i[0][-1] * self.mac_energy,
                    )
                },
            ),
        }
        super().__init__(clock_speed)

    def _phu_matmul_task_para_cycles(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        phu_cycles = (
            math.ceil(math.ceil(num_dot_products / self.num_numtiplex) / self.num_cores)
            * length_dot_products
        )
        return phu_cycles

    def _phu_matmul_task_para_energy(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        return num_dot_products * length_dot_products * self.mac_energy


class CPU(Hardware):
    def __init__(self, clock_speed, num_cores):
        self.num_cores = num_cores
        self.mac_energy = CPU_MAC
        self.algs = {
            "add": HardwareAlgorithm(
                "add",
                {self: (self._add_cycles, energy_per_cycle_func_gen(self._add_cycles))},
            ),
            "subtract": HardwareAlgorithm(
                "subtract", {self: (all_elm, energy_per_cycle_func_gen(all_elm))}
            ),
            "multiply": HardwareAlgorithm(
                "multiply", {self: (all_elm, energy_per_cycle_func_gen(all_elm))}
            ),
            "divide": HardwareAlgorithm(
                "divide", {self: (all_elm, energy_per_cycle_func_gen(all_elm))}
            ),
            "sqrt": HardwareAlgorithm(
                "sqrt", {self: (all_elm, energy_per_cycle_func_gen(all_elm))}
            ),
            "rsqrt": HardwareAlgorithm(
                "rsqrt", {self: (all_elm, energy_per_cycle_func_gen(all_elm))}
            ),
            "relu": HardwareAlgorithm(
                "relu",
                {self: (all_elm_const(2), energy_per_cycle_func_gen(all_elm_const(2)))},
            ),
            "tanh": HardwareAlgorithm(
                "tanh",
                {self: (all_elm_const(4), energy_per_cycle_func_gen(all_elm_const(4)))},
            ),
            "power": HardwareAlgorithm(
                "power", {self: (all_elm, energy_per_cycle_func_gen(all_elm))}
            ),
            "transpose": HardwareAlgorithm(
                "transpose", {self: (all_elm, energy_per_cycle_func_gen(all_elm))}
            ),
            "nop": HardwareAlgorithm(
                "nop", {self: (constnat(1), energy_per_cycle_func_gen(constnat(1)))}
            ),
            "less": HardwareAlgorithm(
                "less", {self: (constnat(1), energy_per_cycle_func_gen(constnat(1)))}
            ),
            "take": HardwareAlgorithm(
                "take", {self: (constnat(1), energy_per_cycle_func_gen(constnat(1)))}
            ),
            "mean": HardwareAlgorithm(
                "mean",
                {
                    self: (
                        self._mean_cycles,
                        energy_per_cycle_func_gen(self._add_cycles),
                    )
                },
            ),
            "softmax": HardwareAlgorithm(
                "softmax",
                {self: (all_elm_const(6), energy_per_cycle_func_gen(all_elm_const(6)))},
            ),
            "matmul": HardwareAlgorithm(
                "matmul", {self: (self._cpu_matmul_cycles, self._cpu_matmul_energy)}
            ),
            "dense": HardwareAlgorithm(
                "dense", {self: (self._cpu_matmul_cycles, self._cpu_matmul_energy)}
            ),
            "pack": HardwareAlgorithm(
                "pack", {self: (self._cpu_matmul_cycles, self._cpu_matmul_energy)}
            ),
            "where": HardwareAlgorithm(
                "where", {self: (constnat(1), energy_per_cycle_func_gen(constnat(1)))}
            ),
            "erf": HardwareAlgorithm(
                "erf", {self: (constnat(1), energy_per_cycle_func_gen(constnat(1)))}
            ),
            "slice": HardwareAlgorithm(
                "slice", {self: (constnat(1), energy_per_cycle_func_gen(constnat(1)))}
            ),
            "negative": HardwareAlgorithm(
                "negative", {self: (all_elm, energy_per_cycle_func_gen(all_elm))}
            ),
            "concatenate": HardwareAlgorithm(
                "concatenate",
                {self: (constnat(1), energy_per_cycle_func_gen(constnat(1)))},
            ),
            "sigmoid": HardwareAlgorithm(
                "sigmoid",
                {self: (all_elm_const(4), energy_per_cycle_func_gen(all_elm_const(4)))},
            ),
            "cast": HardwareAlgorithm(
                "cast", {self: (all_elm, energy_per_cycle_func_gen(all_elm))}
            ),
            "equal": HardwareAlgorithm(
                "equal", {self: (all_elm, energy_per_cycle_func_gen(all_elm))}
            ),
            "to": HardwareAlgorithm(
                "to", {self: (all_elm, energy_per_cycle_func_gen(all_elm))}
            ),
            "nd": HardwareAlgorithm(
                "nd", {self: (all_elm, energy_per_cycle_func_gen(all_elm))}
            ),
        }
        super().__init__(clock_speed)

    def _mean_cycles(self, i, o):
        return (i[0][-1] + 1) * i[0][-2]

    def _add_cycles(self, i, o):
        m = 5.728114536194012e-11
        b = 2.018610633633051e-07

        num_add = all_elm(i, o)

        time = (m * num_add) + b
        return time * self.clock_speed
        """liner fit gives time. Multiply by clock_speed so it gets cancled by
            clock_period durring HardwareAlgorithm.time_cost"""

    def _cpu_matmul_cycles(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        num_mac = num_dot_products * length_dot_products

        if recursive_contains_num(i, lambda x: x > 50000):
            # just tvmgen_default_fused_nn_dense_4
            m = 3.554631597170242e-10
            b = -0.007057115703999989
        else:
            # excluding tvmgen_default_fused_nn_dense_4
            m = 6.845684637179875e-11
            b = 8.258292187675319e-07

        time = (m * num_mac) + b
        return time * self.clock_speed
        """liner fit gives time. Multiply by clock_speed so it gets cancled by
            clock_period durring HardwareAlgorithm.time_cost"""

    def _cpu_matmul_energy(self, i, o):
        num_dot_products = ten_elm(o[0])
        length_dot_products = i[0][-1]
        return num_dot_products * length_dot_products * self.mac_energy


class GPU(Hardware):
    def __init__(self, clock_speed):
        self.mac_energy = GPU_MAC
        self.algs = {}
        super().__init__(clock_speed)


class HBM(Hardware):
    def __init__(self, clock_speed):
        self.algs = {
            "get_dram": HardwareAlgorithm(
                "memory",
                {
                    self: (
                        # sum(ten_elm(a) for a in o) * BITS_PER_NUM / MEMORY_TRANSFER_WIDTH
                        lambda i, o: 0,  # 0 if assuming model is preloaded to HMB
                        lambda i, o: sum(ten_elm(a) for a in o) * DRAM_READ
                        + sum(ten_elm(a) for a in o) * HBM_WRITE,
                    )
                },
            ),
        }
        super().__init__(clock_speed)


class SRAM(Hardware):
    def __init__(self, clock_speed):
        self.algs = {
            "split": HardwareAlgorithm(
                "split", {self: (constnat(1), energy_per_cycle_func_gen(constnat(1)))}
            )
        }
        super().__init__(clock_speed)


class Start(Hardware):
    def __init__(self, clock_speed):
        self.algs = {
            "start": HardwareAlgorithm(
                "start", {self: (constnat(1), energy_per_cycle_func_gen(constnat(1)))}
            ),
        }
        super().__init__(clock_speed)


# endregion

# region constants

NODE_COUNT = 0

# region constants and helpers
MEMORY_TRANSFER_WIDTH = 32  # bits per cycle
DAC_ADC_DELAY = 10 * 10**-9  # 10 nano-seconds

BITS_PER_NUM = 32  # TODO fix to be opp dependant based on the model
MEMORY_CLOCK = 6 * 10**9

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
PHU_MAC = 0.04 * PICO_JOULE
CPU_MAC = 0.1 * PICO_JOULE
GPU_MAC = 0.1 * PICO_JOULE

DAC_POWER = 3.18 * PICO_JOULE
ADC_POWER = 1.6 * PICO_JOULE

# endregion
