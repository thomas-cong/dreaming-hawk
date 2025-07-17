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
    ax.set_title('Word Graph')
    ax.set_xticks([])
    ax.set_yticks([])
    has_semantic = False
    has_temporal = False
    for u, v, data in wg.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        if 'weight' in data:  # Semantic edge
            ax.plot([x1, x2], [y1, y2], color='blue', linestyle='solid', linewidth=1.5, zorder=1)
            # Draw edge weight
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x, mid_y, f"{data['weight']:.2f}", fontsize=7, color='darkgreen', ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))
            has_semantic = True
        elif data.get('type') == 'temporal':
            ax.plot([x1, x2], [y1, y2], color='red', linestyle='dashed', linewidth=1.0, zorder=1)
            has_temporal = True
    # Draw nodes and labels
    if wg.nodes(): # Check if there are nodes to draw
        node_sizes = [wg.nodes[n]['data'].get_value() * 30 for n in wg.nodes()]
        node_x = [pos[n][0] for n in wg.nodes()]
        node_y = [pos[n][1] for n in wg.nodes()]
        ax.scatter(node_x, node_y, s=node_sizes, color='skyblue', zorder=2, ec='black')
    for node, (x, y) in pos.items():
        ax.text(x, y, node, ha='center', va='center', fontsize=2, weight='bold')
    # Create and display legend
    legend_elements = []
    if any('weight' in d for _, _, d in wg.edges(data=True)):
        legend_elements.append(Line2D([0], [0], color='blue', lw=1.5, label='Semantic'))
    if any(d.get('type') == 'temporal' for _, _, d in wg.edges(data=True)):
        legend_elements.append(Line2D([0], [0], color='red', linestyle='dashed', lw=1.0, label='Temporal'))
    
    if legend_elements:
        ax.legend(handles=legend_elements)

def animateGraphBuilding(text_path, window_size, frame_step):
    fig, ax = plt.subplots(figsize=(10, 8))
    wg = WordGraph(text_window_size=window_size)
    text = parse_text(text_path, mode = 'words')
    full_graph_generator = wg.addText(text=text, yield_frames=True, frame_step=frame_step)
    all_frames = list(full_graph_generator)
    if not all_frames:
        print("No frames generated.")
        return
    final_graph = all_frames[-1]
    pos = nx.spring_layout(final_graph, seed=42)
    wg = WordGraph(text_window_size=window_size)
    animation_generator = wg.addText(text=text, yield_frames=True, frame_step=frame_step)
    def update(frame_graph):
        visualizeWordGraph(frame_graph, ax, pos)
    ani = animation.FuncAnimation(fig, update, frames=animation_generator, repeat=False, interval=30, save_count=len(all_frames))
    plt.show()
def main():
    text_path = "/Users/tcong/dreaming-hawk/TrainingTexts/ChalmersPaper.txt"
    window_size = 5
    frame_step = 1  # Adjust this to control animation speed/granularity
    animateGraphBuilding(text_path, window_size, frame_step)
if __name__ == "__main__":
    main()
    