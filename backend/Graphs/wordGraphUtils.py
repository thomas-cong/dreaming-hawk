import os, sys, pathlib
import threading
import tty, termios, time

# Ensure project root on path so that `import dreaming_hawk` works when this file
# is executed directly from its subdirectory.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dreaming_hawk import textUtils, wordGraph as WordGraphModule

# Re-export for brevity
parse_text = textUtils.parse_text
cosine_similarity = textUtils.cosine_similarity
encode_text = textUtils.encode_text
WordGraph = WordGraphModule.WordGraph
import networkx as nx
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from matplotlib.lines import Line2D


def visualizeWordGraph(wg: "WordGraph", ax, pos):
    ax.clear()
    ax.set_title("Word Graph")
    ax.set_xticks([])
    ax.set_yticks([])
    has_semantic = False
    has_temporal = False
    for u, v, data in wg.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        if data.get("type") == "temporal":
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->",
                    color="red",
                    linestyle="dashed",
                    linewidth=1.0,
                ),
                zorder=1,
            )
            has_temporal = True
        elif data.get("type") == "semantic":  # Semantic edge
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


def _precompute_layout(graph: "WordGraph"):
    """Return a stable spring layout for *graph*.
    The spring layout can be slow on large graphs so we pin the random seed to
    ensure deterministic results across runs.
    """
    return nx.spring_layout(graph, seed=42)


def monitor_input_by_word(graph: "WordGraph"):
    """Monitors terminal for new input word-by-word and adds it to the graph."""
    print("Initializing word-level input monitoring. Press Ctrl+C to exit.")

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(sys.stdin.fileno())

        current_word = ""
        previous_word = None

        while True:
            char = sys.stdin.read(1)
            # Handle backspace/delete key
            if char == "\x7f":  # ASCII code for backspace/delete
                if current_word:
                    # Remove last character from the word
                    current_word = current_word[:-1]
                    # Move cursor back, write space to erase, and move back again
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif char.isspace():
                if current_word:
                    if current_word == ":q":
                        print("\n Exiting dreaming-hawk...")
                        break
                    # sys.stdout.write(f'\rWord completed: {current_word}\n')
                    sys.stdout.flush()
                    graph.add_text(current_word)
                    current_word = ""
                sys.stdout.write(char)
                sys.stdout.flush()
            else:
                sys.stdout.write(char)
                sys.stdout.flush()
                current_word += char
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print("\nTerminal settings restored.")


def initializeLiveGraph(graph: "WordGraph"):
    """
    Initializes the live graph and starts monitoring for terminal input
    at the word level.
    """
    with open("./Graphs/WordGraph/hawkASCII.txt", "r") as f:
        for line in f:
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(0.03)
    sys.stdout.write("Welcome to dreaming-hawk CLI v0 \n")
    sys.stdout.write("\n Initializing live graph... \n")
    graph.warm_up()
    input_thread = threading.Thread(
        target=monitor_input_by_word, args=(graph,), daemon=True
    )
    input_thread.start()
    sys.stdout.write("Graph initialized. Start typing to add words to the graph. \n")
    input_thread.join()


def animateGraphBuilding(text_path: str, window_size: int, frame_step: int):
    """Animate graph construction without duplicating work or materialising every frame."""
    with open(text_path, "r") as f:
        text = f.read()

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


def main():
    wg = WordGraph(text_window_size=5)
    with open("/Users/tcong/dreaming-hawk/TrainingTexts/ChalmersPaper.txt", "r") as f:
        text = f.read()
    wg.add_text(text)
    initializeLiveGraph(wg)
    pos = _precompute_layout(wg)
    fig, ax = plt.subplots(figsize=(10, 8))
    visualizeWordGraph(wg, ax, pos)
    print(wg.jsonify())
    plt.show()


if __name__ == "__main__":
    main()
