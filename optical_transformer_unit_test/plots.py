# %%

import json
import matplotlib.pyplot as plt

# energy_filename = "gpt2_prefill_energy_CPU-False_PHU-True_GPU-True.json"
# time_filename = "gpt2_prefill_time_CPU-False_PHU-True_GPU-True.json"
energy_filename = "llama_prefill_energy_CPU-False_PHU-True_GPU-True.json"
time_filename = "llama_prefill_time_CPU-False_PHU-True_GPU-True.json"

# Load data from both files
def load_results(filename):
    with open(filename) as f:
        data = json.load(f)
    results = {int(k): v for k, v in data["results"].items()}
    return dict(sorted(results.items()))

def load_metadata(filename):
    with open(filename) as f:
        data = json.load(f)

    return data["_metadata"]



# %%
#### Makespan plot ####
#### Energy plot ####

energy_metadata = load_metadata(energy_filename)
time_metadata = load_metadata(time_filename)

energy_opt_data = load_results(energy_filename)
time_opt_data = load_results(time_filename)

# Extract aligned data
sequence_lengths = [v["moc_sequence_length"] for v in energy_opt_data.values()]
energy = [v["total_energy"] for v in energy_opt_data.values()]
makespans = [time_opt_data[k]["Makespan"] for k in energy_opt_data.keys()]
macs = [v["num_mac"] for v in energy_opt_data.values()]

energy_opt_makespan = [energy_opt_data[k]["Makespan"] for k in energy_opt_data.keys()]
energy_opt_energy = [energy_opt_data[k]["total_energy"]/ 10**12 for k in energy_opt_data.keys()]
time_opt_makespan = [time_opt_data[k]["Makespan"] for k in time_opt_data.keys()]
time_opt_energy = [time_opt_data[k]["total_energy"]/ 10**12 for k in time_opt_data.keys()]


# AIP-compliant plot styling
plt.rcParams.update({
    "font.family": "serif",       # Serif font for publication quality
    "font.size": 6,               # Base font size
    "axes.labelsize": 6,
    "axes.titlesize": 8,
    "legend.fontsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "lines.linewidth": 0.5,      # At least 0.5 pt line width
    "pdf.fonttype": 42,           # Embed fonts in PDF
    "ps.fonttype": 42
})


# Constants for AIP figure sizing
ONE_COL_WIDTH = 3.37  # inches
TWO_COL_WIDTH = 6.69  # inches
MAX_HEIGHT = 8.25     # inches
MARKER_SIZE = 1.5
LINE_WIDTH = 0.6

# %%
# Makespan plot
def plot_makespan_vs_sequence_length():
    title = f'{energy_metadata["graph_search"]["model"]} Forward Pass'
    fig = plt.figure(figsize=(ONE_COL_WIDTH, 2.5))  # One-column width figure
    plt.plot(sequence_lengths, energy_opt_makespan, marker='o', markersize=MARKER_SIZE,
             linewidth=LINE_WIDTH, label='Energy-Optimized', color='tab:blue')
    plt.plot(sequence_lengths, time_opt_makespan, marker='s', linestyle='--', markersize=MARKER_SIZE,
             linewidth=LINE_WIDTH, label='Time-Optimized', color='tab:orange')
    plt.xlabel("MOC Sequence Length")
    plt.ylabel("Makespan (s)")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}_makespan.png", dpi=300)
    plt.close(fig)

# Energy plot
def plot_energy_vs_sequence_length():
    title = f'{time_metadata["graph_search"]["model"]} Forward Pass'
    fig = plt.figure(figsize=(ONE_COL_WIDTH, 2.5))
    plt.plot(sequence_lengths, energy_opt_energy, marker='o', markersize=MARKER_SIZE,
             linewidth=LINE_WIDTH, label='Energy-Optimized', color='tab:blue')
    plt.plot(sequence_lengths, time_opt_energy, marker='s', linestyle='--', markersize=MARKER_SIZE,
             linewidth=LINE_WIDTH, label='Time-Optimized', color='tab:orange')
    plt.xlabel("MOC Sequence Length")
    plt.ylabel("Total Energy (J)")
    plt.yscale('log')  # For wide energy ranges
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title}_energy.png", dpi=300)
    plt.close(fig)


def makespan_and_energy_stack():
    title = f'{energy_metadata["graph_search"]["model"]} Forward Pass'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(ONE_COL_WIDTH, 4.6), sharex=True)

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
    plt.savefig(f"{title}_combined.png", dpi=300)
    plt.close(fig)


def plot_makespan_and_energy_dual_axis():
    title = f'{energy_metadata["graph_search"]["model"]} Forward Pass'
    fig, ax1 = plt.subplots(figsize=(TWO_COL_WIDTH, 2.5))

    # Plot Makespan on primary y-axis (left)
    ln1 = ax1.plot(sequence_lengths, energy_opt_makespan, marker='o', markersize=MARKER_SIZE,
                   linewidth=LINE_WIDTH, label='Makespan (Energy-Opt)', color='tab:blue')
    ln2 = ax1.plot(sequence_lengths, time_opt_makespan, marker='s', linestyle='--', markersize=MARKER_SIZE,
                   linewidth=LINE_WIDTH, label='Makespan (Time-Opt)', color='tab:orange')
    ax1.set_ylabel("Makespan (s)")
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Energy on secondary y-axis (right)
    ax2 = ax1.twinx()
    ln3 = ax2.plot(sequence_lengths, energy_opt_energy, marker='o', markersize=MARKER_SIZE,
                   linewidth=LINE_WIDTH, label='Energy (Energy-Opt)', color='tab:blue', alpha=0.5)
    ln4 = ax2.plot(sequence_lengths, time_opt_energy, marker='s', linestyle='--', markersize=MARKER_SIZE,
                   linewidth=LINE_WIDTH, label='Energy (Time-Opt)', color='tab:orange', alpha=0.5)
    ax2.set_ylabel("Total Energy (J)")
    ax2.set_yscale('log')

    # Combine legends
    lines = ln1 + ln2 + ln3 + ln4
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='best')

    # Final touches
    ax1.set_xlabel("MOC Sequence Length")
    ax1.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{title}_dual_axis.png", dpi=300)
    plt.close(fig)

# Generate the plot

# Generate combined plot
# plot_makespan_vs_sequence_length()
# plot_energy_vs_sequence_length()
makespan_and_energy_stack()
# plot_makespan_and_energy_dual_axis()



# %%
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(ONE_COL_WIDTH, 4.6), sharex=True)

    # Makespan subplot
    for label in sorted(filenames.keys(), key=int):
        ax1.plot(sequence_lengths, multiplex_makespan[label], marker='o', markersize=MARKER_SIZE,
                 linewidth=LINE_WIDTH, label=f"PHU multiplex {label}")
    ax1.set_ylabel("Makespan (s)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_yscale('log')
    ax1.legend(loc='best', ncol=2)
    ax1.set_title(title)

    # Energy subplot
    for label in sorted(filenames.keys(), key=int):
        ax2.plot(sequence_lengths, multiplex_energy[label], marker='s', markersize=MARKER_SIZE,
                 linewidth=LINE_WIDTH, label=f"PHU multiplex  {label}")
    ax2.set_xlabel("MOC Sequence Length")
    ax2.set_ylabel("Total Energy (J)")
    ax2.set_yscale('log')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='best', ncol=2)

    plt.tight_layout()
    plt.savefig("llama_multiplex_combined.png", dpi=300)
    plt.close(fig)



# Define the files and labels
multiplex_filenames = {
    "1": "llama_prefill_always_phu_CPU-False_PHU-True_1_1_GPU-True.json",
    "5": "llama_prefill_always_phu_CPU-False_PHU-True_1_5_GPU-True.json",
    "10": "llama_prefill_always_phu_CPU-False_PHU-True_1_10_GPU-True.json",
    "15": "llama_prefill_always_phu_CPU-False_PHU-True_1_15_GPU-True.json",
    "20": "llama_prefill_always_phu_CPU-False_PHU-True_1_20_GPU-True.json",
}

# Generate plots
multiplex_makespan_and_energy_stack(multiplex_filenames)

# %% Percentage Compute

import numpy as np

# total, PHU, GPU
data = {
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

    # Extract values
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
fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH, 2.5))
title = "Energy Source Breakdown by Component"

# Stacked bars
ax.bar(x, phu_pct, width=bar_width, label="PHU (%)")
ax.bar(x, gpu_pct, width=bar_width, bottom=phu_pct, label="GPU (%)")
bottom_stack = np.array(phu_pct) + np.array(gpu_pct)
ax.bar(x, other_pct, width=bar_width, bottom=bottom_stack, label="Data Transfer (%)")

# Axes labels and styling
ax.set_xticks(x)
ax.set_xticklabels(sequence_lengths, rotation=90, fontsize=5)  # <-- updated
ax.set_xlabel("Sequence Length")
ax.set_ylabel("Share of Total Energy (%)")
ax.set_title(title)
ax.set_ylim(0, 100)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='upper right', ncol=1)

plt.tight_layout()
plt.savefig("energy_source_breakdown.png", dpi=300)
plt.close(fig)

# %%
