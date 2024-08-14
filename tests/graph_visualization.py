"""
tools for visualizing and drawing graphs and schedules
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import hardware as hw


class GraphVisualization:

    def __init__(self):
        self.visual = []

    def addEdge(self, a, b, weight=1):
        temp = [a, b, weight]
        self.visual.append(temp)

    def normalize(self, value, min_value, max_value):
        assert min_value != max_value
        return max(((value - min_value) / (max_value - min_value)), 0.25)

    def visualize(self, layout="spring", filename="graph.png"):
        G = nx.DiGraph()
        # G.add_edges_from(self.visual)

        weight_range = 0
        for edge in self.visual:
            a, b, weight = edge
            weight_range = max(weight_range, weight)
            G.add_edge(a, b, weight=weight)

        # Choose a layout algorithm
        if layout == "spring":
            # pos = nx.spring_layout(G, k=0.1, iterations=10)
            pos = nx.spring_layout(G, weight="weight", k=0.1, iterations=10)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G, scale=1.0)
        elif layout == "kk":
            lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
            pos = nx.kamada_kawai_layout(G, dist=lengths)
            # pos = nx.kamada_kawai_layout(G)

        else:
            raise ValueError(
                "Unknown layout type. Choose from 'spring', 'shell', or 'spectral'."
            )

        plt.figure(figsize=(15, 15))

        edges = G.edges(data=True)
        weights = [self.normalize(edge[2]["weight"], 0, weight_range) for edge in edges]

        nx.draw(
            G,
            pos,
            node_size=100,
            width=weights,
            with_labels=False,
            node_color="lightblue",
            edge_color="black",
        )

        # Add labels to the nodes
        labels = {node: str(node) for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

        # Save the plot to a file
        plt.savefig(filename, dpi=300)  # Adjust dpi as needed
        plt.close()


def adj_to_graph(
    graph,
    ax,
    save=False,
    layout="shell",
    title="Graph Visualization from Adjacency Matrix",
    stack=False,
):
    vectorized_function = np.vectorize(lambda x: 1 if x is not None else 0)

    adj_matrix = vectorized_function(graph.adj_matrix)

    G = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)

    labels = {}
    colors = {}
    for idx, node in enumerate(G.nodes):
        # labels[node] = graph.node_list[idx].stack_id
        if stack:
            labels[node] = graph.node_list[idx].opp
            colors[node] = "lightblue"
        else:
            labels[node] = graph.node_list[idx].stack.opp

            if isinstance(graph.node_list[idx].get_algo_info("hardware"), hw.PHU):
                colors[node] = "lightcoral"
            else:
                colors[node] = "lightblue"

    nx.set_node_attributes(G, labels, "label")

    if layout == "shell":
        pos = nx.shell_layout(G)
    elif layout == "spectral":
        pos = nx.spectral_layout(G, scale=1.0)
    elif layout == "kk":
        lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        pos = nx.kamada_kawai_layout(G, dist=lengths)
    elif layout == "spring":
        # pos = nx.spring_layout(G, k=0.1, iterations=10)
        pos = nx.spring_layout(G, weight="weight", k=0.3, iterations=10)

    if stack:
        nx.draw(
            G,
            pos,
            with_labels=False,
            node_color=[colors[node] for node in G.nodes],
            edge_color="grey",
            node_size=10,
            width=0.4,
            arrowsize=4,
        )
    else:
        nx.draw(
            G,
            pos,
            with_labels=True,
            labels=labels,
            node_color=[colors[node] for node in G.nodes],
            edge_color="grey",
            node_size=400,
            font_size=10,
            # ax=ax,
        )
        ax.set_title(title)
        ax.set_aspect("equal")


def make_schedule_diagram(graph, xlim_start=None, xlim_end=None):
    data = {"task": [], "start": [], "end": [], "label": []}  # Labels for the blocks
    hardware = set()
    for node in graph.node_list:
        data["task"].append(node.hardware_selection)
        hardware.add(node.hardware_selection)
        data["start"].append(node.start_time)
        data["end"].append(node.start_time + node.time_cost)
        data["label"].append(node.stack_id)

    df = pd.DataFrame(data)
    print(df.shape)

    if xlim_start is not None and xlim_end is not None:
        df = df[(df["start"] >= xlim_start) & (df["end"] <= xlim_end)]

    unique_combinations = df[["task", "label"]].drop_duplicates()
    colors = plt.cm.get_cmap("tab10", len(unique_combinations))

    color_map = {}
    for idx, (task, label) in enumerate(unique_combinations.itertuples(index=False)):
        color_map[(task, label)] = colors(idx)

    fig, ax = plt.subplots(figsize=(60, 4))

    # Iterate over each row in the dataframe and plot a horizontal line
    for index, row in df.iterrows():
        task_label = (row["task"], row["label"])
        task_color = color_map[task_label]
        ax.hlines(
            y=row["task"],
            xmin=row["start"],
            xmax=row["end"],
            color=task_color,
            linewidth=25,
        )

        # Adding label on the bar
        # label_text = f"{row['label']}"
        # ax.text((row['start'] + row['end']) / 2, row['task'], label_text, color='black', ha='center', va='center')

    # Customize the plot
    ax.set_xlabel("Time")
    ax.set_ylabel("Task")
    if xlim_start and xlim_end:
        ax.set_title(f"Task Schedule {round(xlim_start, 5)}-{round(xlim_end, 5)}")
    else:
        ax.set_title(f"Task Schedule all")
    ax.grid(False)
    # ax.legend()

    # x_ticks = np.arange(start=float(xlim_start), stop=float(xlim_end), step=float(1e-8 / 10))
    # ax.set_xticks(x_ticks)

    # if xlim_start is not None and xlim_end is not None:
    #     ax.set_xlim(xlim_start, xlim_end)

    plt.show()
