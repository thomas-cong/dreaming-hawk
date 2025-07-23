import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from textUtils import parse_text, cosine_similarity, encode_text
from wordGraph import WordGraph
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

import matplotlib.animation as animation
from matplotlib.lines import Line2D


def visualizeWordGraph(wg, ax, pos):
    ax.clear()
    ax.set_title("Word Graph")
    ax.set_xticks([])
    ax.set_yticks([])
    has_semantic = False
    has_temporal = False
    for u, v, data in wg.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        if "weight" in data:  # Semantic edge
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="blue",
                linestyle="solid",
                linewidth=1.5,
                zorder=1,
            )
            # Draw edge weight
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(
                mid_x,
                mid_y,
                f"{data['weight']:.2f}",
                fontsize=7,
                color="darkgreen",
                ha="center",
                va="center",
                bbox=dict(
                    facecolor="white",
                    alpha=0.5,
                    edgecolor="none",
                    boxstyle="round,pad=0.1",
                ),
            )
            has_semantic = True
        elif data.get("type") == "temporal":
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="red",
                linestyle="dashed",
                linewidth=1.0,
                zorder=1,
            )
            has_temporal = True
    # Draw nodes and labels
    if wg.nodes():  # Check if there are nodes to draw
        node_sizes = [wg.nodes[n]["data"].get_value() * 30 for n in wg.nodes()]
        node_x = [pos[n][0] for n in wg.nodes()]
        node_y = [pos[n][1] for n in wg.nodes()]
        ax.scatter(node_x, node_y, s=node_sizes, color="skyblue", zorder=2, ec="black")
    for node in wg.nodes():
        x, y = pos[node]
        # Ensure font remains readable but proportional to node value
        fontsize = max(6, wg.nodes[node]["data"].get_value() + 4)
        ax.text(x, y, node, ha="center", va="center", fontsize=fontsize, weight="bold")
    # Create and display legend
    legend_elements = []
    if any("weight" in d for _, _, d in wg.edges(data=True)):
        legend_elements.append(Line2D([0], [0], color="blue", lw=1.5, label="Semantic"))
    if any(d.get("type") == "temporal" for _, _, d in wg.edges(data=True)):
        legend_elements.append(
            Line2D([0], [0], color="red", linestyle="dashed", lw=1.0, label="Temporal")
        )

    if legend_elements:
        ax.legend(handles=legend_elements)


def _precompute_layout(graph: WordGraph):
    """Return a stable spring layout for *graph*.
    The spring layout can be slow on large graphs so we pin the random seed to
    ensure deterministic results across runs.
    """
    return nx.spring_layout(graph, seed=42)


def animateGraphBuilding(text_path: str, window_size: int, frame_step: int):
    """Animate graph construction without duplicating work or materialising every frame."""
    text = parse_text(text_path, mode="words")

    # Build *one* graph to obtain final structure for layout computation – this
    # is O(N) once, instead of twice as before.
    wg_final = WordGraph(text_window_size=window_size)
    wg_final.add_text(text=text, yield_frames=False)
    pos = _precompute_layout(wg_final)

    # Create a fresh graph for the animation stream.
    wg_anim = WordGraph(text_window_size=window_size)
    frame_gen = wg_anim.add_text(text=text, yield_frames=True, frame_step=frame_step)

    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame_graph):
        visualizeWordGraph(frame_graph, ax, pos)

    ani = animation.FuncAnimation(
        fig, update, frames=frame_gen, repeat=False, interval=30
    )
    plt.show()
    return wg_final


def animateGraphEnrichment(text_path: str, frame_step: int, wg: WordGraph):
    """Animate semantic-edge enrichment while avoiding redundant passes."""
    with open(text_path, "r") as f:
        text = f.read()

    # Compute final enriched graph once for layout.
    wg_enriched = wg.copy()
    wg_enriched.enrich_semantic_connections(text=text, yield_frames=False)
    pos = _precompute_layout(wg_enriched)

    # Stream enrichment frames from a *fresh* copy (so initial state is clean).
    wg_stream = wg.copy()
    frame_gen = wg_stream.enrich_semantic_connections(
        text=text, yield_frames=True, frame_step=frame_step
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame_graph):
        visualizeWordGraph(frame_graph, ax, pos)

    ani = animation.FuncAnimation(
        fig, update, frames=frame_gen, repeat=False, interval=10
    )
    plt.show()


def main():
    animateGraphBuilding(text_path="./TrainingTexts/ChalmersPaper.txt", window_size=5, frame_step=1)
if __name__ == "__main__":
    main()
