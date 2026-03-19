import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
import matplotlib.text as mtext

C_DEFAULT  = "#90caf9" # unvisited node 
C_EXPANDED = "#1565c0" # expanded node  
C_ORIGIN   = "#4caf50" # origin         
C_GOAL     = "#ef5350" # destination    
C_PATH     = "#ff9800" # final path     
C_EDGE     = "#aaaaaa" # normal edge
C_PEDGE    = "#e63946" # path edge      

NODE_SIZE = 600
FONT_SIZE = 10
ANIM_MS   = 700 # milliseconds per step

# draw an edge with optional arrow for directed graphs
def _draw_edge(ax, x1, y1, x2, y2, color, lw, directed):
    if directed:
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color,
                            lw=lw, mutation_scale=14),
            zorder=2
        )
    else:
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, zorder=1)

# main visualization function
def launch(nodes, edges, origin, destinations, final_path, expanded_order, test_case_number=None, method=None):
    dest_set   = set(destinations)
    path_set   = set(final_path) if final_path else set()
    path_edges = set(zip(final_path, final_path[1:])) if final_path else set()

    xs = {n: c[0] for n, c in nodes.items()}
    ys = {n: c[1] for n, c in nodes.items()}

    def is_one_way(frm, to):
        return frm not in edges.get(to, [])

    # figure setup 
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#f5f5f5")
    ax.set_facecolor("white")
    # set title and window name
    if test_case_number is not None and method is not None:
        title_str = f"Test Case {test_case_number} - {method}"
        ax.set_title(title_str, fontsize=12, fontweight="bold", pad=10)
        try:
            fig.canvas.manager.set_window_title(title_str)
        except Exception:
            pass
    else:
        ax.set_title("Graph Visualizer", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, color="#e0e0e0", linewidth=0.8)
    ax.set_axisbelow(True)

    pad = 1
    ax.set_xlim(min(xs.values()) - pad, max(xs.values()) + pad)
    ax.set_ylim(min(ys.values()) - pad, max(ys.values()) + pad)

    status_text = ax.text(
        0.5, -0.12, "Animating...",
        transform=ax.transAxes, ha="center", va="top",
        fontsize=10, color="#444444"
    )

    legend_items = [
        mpatches.Patch(color=C_ORIGIN,   label="Origin"),
        mpatches.Patch(color=C_GOAL,     label="Destination"),
        mpatches.Patch(color=C_EXPANDED, label="Expanded"),
        mpatches.Patch(color=C_PATH,     label="Final path"),
    ]
    ax.legend(handles=legend_items, loc="upper left",
              fontsize=8, framealpha=0.9, edgecolor="#cccccc")

    # drawing helpers 
    scatter_refs = {}
    label_refs   = {}

    def redraw_edges(show_path=False):
        for line in list(ax.lines):
            line.remove()
        # correctly remove annotation arrows using matplotlib.text.Annotation
        for child in list(ax.get_children()):
            if isinstance(child, mtext.Annotation):
                try:
                    child.remove()
                except Exception:
                    pass
        for frm, neighbours in edges.items():
            for to in neighbours:
                x1, y1 = xs[frm], ys[frm]
                x2, y2 = xs[to],  ys[to]
                on_path = show_path and (frm, to) in path_edges
                color   = C_PEDGE if on_path else C_EDGE
                lw      = 2.5     if on_path else 1.2
                _draw_edge(ax, x1, y1, x2, y2, color, lw, is_one_way(frm, to))

    # redraw nodes with appropriate colors and sizes
    def redraw_nodes(visited=None, current=None, show_path=False):
        visited = visited or set()
        for nid in nodes:
            x, y = xs[nid], ys[nid]
            if nid == origin:
                fill = C_ORIGIN
            elif nid in dest_set:
                fill = C_GOAL
            elif show_path and nid in path_set:
                fill = C_PATH
            elif nid in visited:
                fill = C_EXPANDED
            else:
                fill = C_DEFAULT

            edge_c = "#ff6f00" if nid == current else (
                     "#1a237e" if nid in visited else "#555555")
            lw = 2.5 if (nid == current or nid in visited) else 1.0

            if nid in scatter_refs:
                scatter_refs[nid].remove()
            if nid in label_refs:
                label_refs[nid].remove()

            scatter_refs[nid] = ax.scatter(
                x, y, s=NODE_SIZE, c=fill,
                edgecolors=edge_c, linewidths=lw, zorder=3
            )
            label_refs[nid] = ax.text(
                x, y, str(nid),
                ha="center", va="center",
                fontsize=FONT_SIZE, fontweight="bold",
                color="#1a1a1a", zorder=4
            )

    # initial draw
    redraw_edges()
    redraw_nodes()
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    # animation
    visited_so_far = []
    total = len(expanded_order)

    def animate(frame):
        if frame < total:
            node = expanded_order[frame]
            visited_so_far.append(node)
            redraw_edges()
            redraw_nodes(visited=set(visited_so_far), current=node)
            status_text.set_text(
                f"\n\nExpanding Node {node} ({frame + 1} / {total})"
            )
            status_text.set_fontsize(10)
        else:
            redraw_edges(show_path=True)
            redraw_nodes(visited=set(visited_so_far), show_path=True)
            if final_path:
                status_text.set_text(
                    f"\n\nPath: {' -> '.join(map(str, final_path))}"
                )
                status_text.set_fontsize(10)
            else:
                status_text.set_text("\n\nSearch complete — no path found.")
                status_text.set_fontsize(10)

    # create animation         
    anim = animation.FuncAnimation(
        fig,
        animate,
        frames=total + 1,
        interval=ANIM_MS,
        repeat=False
    )

    plt.show()