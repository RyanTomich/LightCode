import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ScalarFormatter


import numpy as np

energy_split_data = {
    0: [0.07278225327660001, 0.0015860760576000342, 0.05393923047100024],
    100: [0.1065828802298, 0.02653316710400016, 0.054506988685000236],
    200: [0.1435966131938, 0.053276049408000646, 0.05525120998500025],
    300: [0.1816986133578, 0.08022864691200013, 0.056139991285000294],
    400: [0.2208888807218, 0.10739095961600185, 0.05717333258500029],
    500: [0.2611674152858, 0.13476298752000462, 0.05835123388500039],
    600: [0.3025342170498, 0.16234473062399804, 0.059673695185000514],
    700: [0.34498928601379997, 0.19013618892799392, 0.061140716485000565],
    800: [0.3885326221778, 0.2181373624320019, 0.06275229778500065],
    900: [0.4331642255418, 0.2463482511359974, 0.06450843908500052],
    1000: [0.47888409610579996, 0.2747688550399991, 0.06640914038500077],
    1100: [0.5256922338698, 0.30339917414400575, 0.0684544016850007],
    1200: [0.5735886388338001, 0.3322392084479999, 0.07064422298500095],
    1300: [0.6225733109977, 0.36128895795200905, 0.07297860428500065],
    1400: [0.6726462503617, 0.39054842265598966, 0.07545754558500037],
    1500: [0.7238074569258001, 0.42001760255998166, 0.07808104688499999],
    1600: [0.7760569306898001, 0.44969649766397646, 0.08084910818500061],
    1700: [0.8293946716537, 0.4795851079679899, 0.08376172948500066],
    1800: [0.8838206798178001, 0.5096834334719906, 0.08681891078500002],
    1900: [0.9393349551818001, 0.5399914741759745, 0.09002065208500046],
    2000: [0.9959374977458001, 0.5705092300799841, 0.09336695338499985],
    2100: [1.0536283075097, 0.6012367011840034, 0.0968578146850003],
    2200: [1.1124073844738, 0.6321738874880198, 0.10049323598499996],
    2300: [1.1722747286378001, 0.6633207889920222, 0.1042732172850003],
    2400: [1.2332303400018, 0.6946774056959883, 0.10819775858500051],
    2500: [1.2952742185658999, 0.7262437376000038, 0.11226685988500075],
    2600: [1.3584063643296, 0.7580197847040078, 0.11648052118500082],
    2700: [1.4226267772937, 0.7900055470079772, 0.12083874248500154],
    2800: [1.4879354574577, 0.8222010245120124, 0.1253415237849999],
    2900: [1.5543324048218001, 0.8546062172160086, 0.12998886508499988],
    3000: [1.6218176193858, 0.887221125119958, 0.1347807663850003],
    3100: [1.6903911011498, 0.920045748224021, 0.1397172276850016],
    3200: [1.7600528501139, 0.9530800865280061, 0.14479824898500185],
    3300: [1.8308028662779, 0.9863241400320087, 0.15002383028500207],
    3400: [1.9026411496416, 1.0197779087359664, 0.155393971585003],
    3500: [1.9755677002057, 1.0534413926399744, 0.16090867288500196],
    3600: [2.0495825179698, 1.0873145917439706, 0.1665679341850032],
    3700: [2.1246856029338, 1.1213975060480044, 0.1723717554850032],
    3800: [2.2008769550977, 1.1556901355519886, 0.17832013678500336],
    3900: [2.2781565744619, 1.1901924802560013, 0.18441307808499977],
}


def makespan_and_energy_stack_all(
        all_data,
        all_metadata,
        all_sequence_lengths
):
    fig, axes = plt.subplots(2, 2, figsize=(ONE_COL_WIDTH, ONE_COL_WIDTH))

    for idx, (metadata, sequence_lengths, energy_mks, time_mks, energy_eng, time_eng) in enumerate(zip(
        all_metadata, all_sequence_lengths,
        [d[0] for d in all_data], [d[1] for d in all_data],
        [d[2] for d in all_data], [d[3] for d in all_data]
    )):
        model = metadata["graph_search"]["model"]
        col = idx  # GPT=0 → left; LLaMA=1 → right

        # Makespan (top row)
        ax_mks = axes[0, col]
        ax_mks.plot(sequence_lengths, energy_mks, marker='o', markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH, label='Energy-Optimized', color='tab:blue')
        ax_mks.plot(sequence_lengths, time_mks, marker='s', linestyle='--', markersize=MARKER_SIZE,
                    linewidth=LINE_WIDTH, label='Time-Optimized', color='tab:orange')
        ax_mks.set_title(f"{model}")
        ax_mks.grid(True, linestyle='--', alpha=0.6)
        ax_mks.legend(loc='best')

        # Energy (bottom row)
        ax_energy = axes[1, col]
        ax_energy.plot(sequence_lengths, energy_eng, marker='o', markersize=MARKER_SIZE,
                       linewidth=LINE_WIDTH, label='Energy-Optimized', color='tab:blue')
        ax_energy.plot(sequence_lengths, time_eng, marker='s', linestyle='--', markersize=MARKER_SIZE,
                       linewidth=LINE_WIDTH, label='Time-Optimized', color='tab:orange')

        # ax_energy.set_yscale('log')
        ax_energy.grid(True, linestyle='--', alpha=0.6)
        ax_energy.legend(loc='best')
        ax_energy.set_xlabel("MOC Sequence Length")

        if col ==0:
            ax_mks.set_ylabel("Makespan (s)")
            ax_energy.set_ylabel("Energy (J)")

    plt.tight_layout()
    plt.savefig("makespan_energy_combined.png", dpi=DPI)
    plt.close(fig)


# Load and prepare data
def process_all_models():
    gpt_energy_filename = "gpt2_prefill_energy_CPU-False_PHU-True_GPU-True.json"
    gpt_time_filename = "gpt2_prefill_time_CPU-False_PHU-True_GPU-True.json"
    llama_energy_filename = "llama_prefill_energy_CPU-False_PHU-True_GPU-True.json"
    llama_time_filename = "llama_prefill_time_CPU-False_PHU-True_GPU-True.json"

    files = [(gpt_energy_filename, gpt_time_filename), (llama_energy_filename, llama_time_filename)]

    all_data = []
    all_metadata = []
    all_sequence_lengths = []

    for energy_filename, time_filename in files:
        energy_metadata = load_metadata(energy_filename)
        time_metadata = load_metadata(time_filename)

        energy_opt_data = load_results(energy_filename)
        time_opt_data = load_results(time_filename)

        sequence_lengths = [v["moc_sequence_length"] for v in energy_opt_data.values()]
        if 'gpt' in energy_filename:
            sequence_lengths = [x for x in sequence_lengths if x <= 1024]

        energy_opt_makespan = [energy_opt_data[k]["Makespan"] for k in energy_opt_data.keys()]
        energy_opt_energy = [energy_opt_data[k]["total_energy"] / 1e12 for k in energy_opt_data.keys()]
        time_opt_makespan = [time_opt_data[k]["Makespan"] for k in time_opt_data.keys()]
        time_opt_energy = [time_opt_data[k]["total_energy"] / 1e12 for k in time_opt_data.keys()]

        # Trim to match sequence_lengths
        energy_opt_makespan = energy_opt_makespan[:len(sequence_lengths)]
        energy_opt_energy = energy_opt_energy[:len(sequence_lengths)]
        time_opt_makespan = time_opt_makespan[:len(sequence_lengths)]
        time_opt_energy = time_opt_energy[:len(sequence_lengths)]

        all_data.append((energy_opt_makespan, time_opt_makespan, energy_opt_energy, time_opt_energy))
        all_metadata.append(energy_metadata)
        all_sequence_lengths.append(sequence_lengths)

    makespan_and_energy_stack_all(all_data, all_metadata, all_sequence_lengths)




def makespan_and_energy_stack(
        energy_metadata,
        sequence_lengths,
        energy_opt_makespan,
        time_opt_makespan,
        energy_opt_energy,
        time_opt_energy,
):
    title = f'{energy_metadata["graph_search"]["model"]} Forward Pass'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(ONE_COL_WIDTH, ONE_COL_WIDTH*1.5), sharex=True)

    # Makespan subplot
    ax1.plot(sequence_lengths, energy_opt_makespan, marker='o', markersize=MARKER_SIZE,
             linewidth=LINE_WIDTH, label='Energy-Optimized', color='tab:blue')
    ax1.plot(sequence_lengths, time_opt_makespan, marker='s', linestyle='--', markersize=MARKER_SIZE,
             linewidth=LINE_WIDTH, label='Time-Optimized', color='tab:orange')
    ax1.set_ylabel("Makespan (s)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='best')
    ax1.set_title(title)

    # Energy subplot
    ax2.plot(sequence_lengths, energy_opt_energy, marker='o', markersize=MARKER_SIZE,
             linewidth=LINE_WIDTH, label='Energy-Optimized', color='tab:blue')
    ax2.plot(sequence_lengths, time_opt_energy, marker='s', linestyle='--', markersize=MARKER_SIZE,
             linewidth=LINE_WIDTH, label='Time-Optimized', color='tab:orange')
    ax2.set_xlabel("MOC Sequence Length")
    ax2.set_ylabel("Total Energy (J)")
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.savefig(f"{title}_combined.png", dpi=DPI)
    plt.close(fig)

def load_metadata(filename):
    with open(filename) as f:
        data = json.load(f)

    return data["_metadata"]

def load_results(filename):
    with open(filename) as f:
        data = json.load(f)
    results = {int(k): v for k, v in data["results"].items()}
    return dict(sorted(results.items()))

def multiplex_makespan_and_energy_stack(filenames):
    multiplex_energy = {}
    multiplex_makespan = {}
    sequence_lengths = None

    for label, filename in filenames.items():
        with open(filename) as f:
            data = json.load(f)
        results = {int(k): v for k, v in data["results"].items()}
        results = dict(sorted(results.items()))

        if sequence_lengths is None:
            sequence_lengths = [v["moc_sequence_length"] for v in results.values()]

        multiplex_energy[label] = [v["total_energy"] / 1e12 for v in results.values()]  # pJ -> J
        multiplex_makespan[label] = [v["Makespan"] for v in results.values()]

    title = "LLAMA Prefill Forward Pass Multiplexing"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(ONE_COL_WIDTH, ONE_COL_WIDTH*1.5), sharex=True)

    # Makespan subplot
    for label in sorted(filenames.keys(), key=int):
        ax1.plot(sequence_lengths, multiplex_makespan[label], marker='o', markersize=MARKER_SIZE,
                 linewidth=LINE_WIDTH, label=f"PTU multiplex {label}")
    ax1.set_ylabel("Makespan (s)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_yscale('log')
    ax1.legend(loc='best', ncol=2)
    ax1.set_title(title)

    # Energy subplot
    for label in sorted(filenames.keys(), key=int):
        ax2.plot(sequence_lengths, multiplex_energy[label], marker='s', markersize=MARKER_SIZE,
                 linewidth=LINE_WIDTH, label=f"PTU multiplex  {label}")
    ax2.set_xlabel("MOC Sequence Length")
    ax2.set_ylabel("Total Energy (J)")
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='best', ncol=2)

    plt.tight_layout()
    plt.savefig("llama_multiplex_combined.png", dpi=DPI)
    plt.close(fig)

def multiplex_makespan_and_energy_side_by_side(filenames):
    multiplex_energy = {}
    multiplex_makespan = {}
    sequence_lengths = None

    for label, filename in filenames.items():
        with open(filename) as f:
            data = json.load(f)
        results = {int(k): v for k, v in data["results"].items()}
        results = dict(sorted(results.items()))

        if sequence_lengths is None:
            sequence_lengths = [v["moc_sequence_length"] for v in results.values()]

        multiplex_energy[label] = [v["total_energy"] / 1e12 for v in results.values()]  # pJ -> J
        multiplex_makespan[label] = [v["Makespan"] for v in results.values()]

    title = "LLAMA Prefill Forward Pass Multiplexing"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(ONE_COL_WIDTH, ONE_COL_WIDTH), sharex=True)

    # Makespan subplot
    for label in sorted(filenames.keys(), key=int):
        ax1.plot(sequence_lengths, multiplex_makespan[label], marker='o', markersize=MARKER_SIZE,
                 linewidth=LINE_WIDTH, label=f"PTU multiplex {label}")
    ax1.set_ylabel("Makespan (s)")
    ax1.set_xlabel("MOC Sequence Length")
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='best', ncol=1)
    ax1.set_title("Makespan")

    # Energy subplot
    for label in sorted(filenames.keys(), key=int):
        ax2.plot(sequence_lengths, multiplex_energy[label], marker='s', markersize=MARKER_SIZE,
                 linewidth=LINE_WIDTH, label=f"PTU multiplex {label}")
    ax2.set_ylabel("Total Energy (J)")
    ax2.set_xlabel("MOC Sequence Length")
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='best', ncol=1)
    ax2.set_title("Energy")

    fig.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("llama_multiplex_combined_sbs.png", dpi=DPI)
    plt.close(fig)


def energy_split(data):
    sequence_lengths = list(data.keys())
    totals = [v[0] for v in data.values()]
    phu_vals = [v[1] for v in data.values()]
    gpu_vals = [v[2] for v in data.values()]
    other_vals = [totals[i] - phu_vals[i] - gpu_vals[i] for i in range(len(totals))]

    # Percentages
    phu_pct = [100 * phu_vals[i] / totals[i] for i in range(len(totals))]
    gpu_pct = [100 * gpu_vals[i] / totals[i] for i in range(len(totals))]
    other_pct = [100 * other_vals[i] / totals[i] for i in range(len(totals))]


    x = np.arange(len(sequence_lengths))
    bar_width = 1

    # Create figure
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, ONE_COL_WIDTH))
    title = "Energy Consumption by Hardware"

    # Stacked bars
    ax.bar(x, phu_pct, width=bar_width, label="PTU (%)")
    ax.bar(x, gpu_pct, width=bar_width, bottom=phu_pct, label="GPU (%)")
    bottom_stack = np.array(phu_pct) + np.array(gpu_pct)
    ax.bar(x, other_pct, width=bar_width, bottom=bottom_stack, label="Data Transfer (%)")

    # Axes labels and styling
    ax.set_xticks(x)
    ax.set_xticklabels(sequence_lengths, rotation=90, fontsize=5)
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Share of Total Energy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', ncol=1)

    plt.tight_layout()
    plt.savefig("energy_source_breakdown.png", dpi=DPI)
    plt.close(fig)

def sensitivity(path):
    with open(path, "r") as f:
        all_data = json.load(f)

    # Configurations
    if 'max' in path:
        models = [("gpt2_prefill", 1024), ("llama_prefill", 4096)]
    else:
        models = [("gpt2_prefill", 100), ("llama_prefill", 100)]

    optimizations = ["time", "energy"]
    params = [
        'PHU_MIN_CLOCK', 'GPU_FP32_CLOCK', 'MEMORY_CLOCK',
        'DRAM_RW_COST', 'SRAM_RW_COST', 'LOCAL_RW_COST',
        'PHU_MAC', 'GPU_MAC', 'DAC_POWER', 'ADC_POWER'
    ]

    # Colors and markers
    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', '*', 'X']

    # Create figure with 4 rows (2 per model) and 2 columns (top: Makespan, bottom: Energy)
    fig, axes = plt.subplots(
        4, 2, figsize=(ONE_COL_WIDTH, ONE_COL_WIDTH*2), sharex='col',
        gridspec_kw={"height_ratios": [1, 1, 1, 1]}
    )
    # fig.subplots_adjust(hspace=0.01, wspace=0.01)

    seen = set()
    handles = []


    axes = axes.reshape(4, 2)

    for plot_idx, (model_name, seq_len) in enumerate(models):
        for opt_idx, optimization in enumerate(optimizations):
            row = plot_idx * 2 + opt_idx
            ax_left, ax_right = axes[row]

            all_percent_labels = None

            for idx, param in enumerate(params):
                key = f"{model_name}_{seq_len}_{optimization}_{param}"
                if key not in all_data:
                    print(f"Missing key: {key}")
                    continue

                data = all_data[key]
                keys = sorted(data.keys(), key=lambda x: float(x))
                x_vals = [float(k) for k in keys]
                makespans = [data[k]["Makespan"] for k in keys]
                # energies = [data[k]["total_energy"] for k in keys]
                energies = [data[k]["total_energy"] / 10**12 for k in keys]
                num_photonic = [data[k]["num_photonic"] for k in keys]

                if any([i<0 for i in makespans]):
                    assert False
                if any([i<0 for i in energies]):
                    assert False

                tolerance = 0
                base_makespan = makespans[0]
                base_energy = energies[0]

                if all(abs(m - base_makespan) / base_makespan <= tolerance for m in makespans) and \
                all(abs(e - base_energy) / base_energy <= tolerance for e in energies):
                    continue


                base = x_vals[len(x_vals) // 2]
                percent_labels = [f"{(v - base) / base * 100:.0f}%" for v in x_vals]
                if all_percent_labels is None:
                    all_percent_labels = percent_labels

                color = colors[idx % len(colors)]
                marker = markers[idx % len(markers)]

                if param == "PHU_MIN_CLOCK":
                    param  = "PTU_MIN_CLOCK"
                if param == "PHU_MAC":
                    param  = "PTU_MAC"

                h, = ax_left.plot(percent_labels, makespans, marker=marker, markersize=MARKER_SIZE*2, label=param, color=color, alpha=0.5)
                ax_right.plot(percent_labels, energies, marker=marker, markersize=MARKER_SIZE*2, color=color, alpha=0.5)
                if param not in seen:
                    handles.append(h)
                    seen.add(param)


            ax_left.set_ylabel("Makespan (s)")
            ax_right.set_ylabel("Energy (J)")

            ax_left.tick_params(axis='both')
            ax_right.tick_params(axis='both')


            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 2))

            ax_left.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax_right.yaxis.set_major_formatter(formatter)

            # ax_right.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))


            # Only label x-axis for the bottom-most row
            if row == 3:
                ax_left.set_xlabel("Parameter Change (% from baseline)", x=1)

            ax_left.grid(True)
            ax_right.grid(True)

            # Titles for each subfigure
            ax_left.set_title(f"{model_name.replace('_', ' ')}, {optimization.capitalize()} Optimization", x=0.85, pad=10)

            # ax_left.set_yscale('log')
            # ax_right.set_yscale('log')



    # Global legend
    fig.legend(
        handles, [h.get_label() for h in handles],
        loc="upper center", bbox_to_anchor=(0.5, 0.96),
        ncol=3, frameon=True
    )

    if 'max' in path:
        fig.suptitle(f"Sensitivity Analysis (max Sequence Length)", fontsize = 10)
    else:
        fig.suptitle(f"Sensitivity Analysis (100 Sequence Length)", fontsize = 10)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("sensitivity_analysis_combined.png", dpi=DPI)
    plt.show()

# AIP-compliant plot styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 6,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "legend.fontsize": 5,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "lines.linewidth": 0.5,
    "ps.fonttype": 42
})

ONE_COL_WIDTH = 3.37
MAX_HEIGHT = 8.25
MARKER_SIZE = 2
LINE_WIDTH = 0.6
DPI = 500



def main():
    gpt_energy_filename = "gpt2_prefill_energy_CPU-False_PHU-True_GPU-True.json"
    gpt_time_filename = "gpt2_prefill_time_CPU-False_PHU-True_GPU-True.json"
    llama_energy_filename = "llama_prefill_energy_CPU-False_PHU-True_GPU-True.json"
    llama_time_filename = "llama_prefill_time_CPU-False_PHU-True_GPU-True.json"

    files = [(gpt_energy_filename, gpt_time_filename), (llama_energy_filename, llama_time_filename)]

    for energy_filename, time_filename in files:


        energy_metadata = load_metadata(energy_filename)
        time_metadata = load_metadata(time_filename)

        energy_opt_data = load_results(energy_filename)
        time_opt_data = load_results(time_filename)

        sequence_lengths = [v["moc_sequence_length"] for v in energy_opt_data.values()]

        if 'gpt' in energy_filename:
            sequence_lengths = [x for x in sequence_lengths if x <= 1024]

        # energy = [v["total_energy"] for v in energy_opt_data.values()]
        # makespans = [time_opt_data[k]["Makespan"] for k in energy_opt_data.keys()]
        # macs = [v["num_mac"] for v in energy_opt_data.values()]

        energy_opt_makespan = [energy_opt_data[k]["Makespan"] for k in energy_opt_data.keys()]
        energy_opt_energy = [energy_opt_data[k]["total_energy"]/ 10**12 for k in energy_opt_data.keys()]
        time_opt_makespan = [time_opt_data[k]["Makespan"] for k in time_opt_data.keys()]
        time_opt_energy = [time_opt_data[k]["total_energy"]/ 10**12 for k in time_opt_data.keys()]

        energy_opt_makespan = energy_opt_makespan[0: len(sequence_lengths)]
        energy_opt_energy = energy_opt_energy[0: len(sequence_lengths)]
        time_opt_makespan = time_opt_makespan[0: len(sequence_lengths)]
        time_opt_energy = time_opt_energy[0: len(sequence_lengths)]

        makespan_and_energy_stack(
            energy_metadata,
            sequence_lengths,
            energy_opt_makespan,
            time_opt_makespan,
            energy_opt_energy,
            time_opt_energy,
        )

    process_all_models()

    multiplex_filenames = {
        "1": "llama_prefill_always_phu_CPU-False_PHU-True_1_1_GPU-True.json",
        "5": "llama_prefill_always_phu_CPU-False_PHU-True_1_5_GPU-True.json",
        "10": "llama_prefill_always_phu_CPU-False_PHU-True_1_10_GPU-True.json",
        "15": "llama_prefill_always_phu_CPU-False_PHU-True_1_15_GPU-True.json",
        "20": "llama_prefill_always_phu_CPU-False_PHU-True_1_20_GPU-True.json",
    }
    multiplex_makespan_and_energy_stack(multiplex_filenames)

    energy_split(energy_split_data)

    path = "sensitivity_results_100.json"
    # path = "sensitivity_results_max.json"
    sensitivity(path)


if __name__ == "__main__":
    pass
    main()
    path = "sensitivity_results_100.json"
    # path = "sensitivity_results_max.json"
    sensitivity(path)
    process_all_models()


    # multiplex_filenames = {
    #     "1": "llama_prefill_always_phu_CPU-False_PHU-True_1_1_GPU-True.json",
    #     "5": "llama_prefill_always_phu_CPU-False_PHU-True_1_5_GPU-True.json",
    #     "10": "llama_prefill_always_phu_CPU-False_PHU-True_1_10_GPU-True.json",
    #     "15": "llama_prefill_always_phu_CPU-False_PHU-True_1_15_GPU-True.json",
    #     "20": "llama_prefill_always_phu_CPU-False_PHU-True_1_20_GPU-True.json",
    # }
    # multiplex_makespan_and_energy_side_by_side(multiplex_filenames)
