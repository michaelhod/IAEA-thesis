from typing import Sequence, Tuple, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap


def _truncate(text: str, head: int = 7, tail: int = 7) -> str:
    s = "" if text is None else str(text)
    if len(s) <= head + tail + 3:
        return s
    return f"{s[:head]}...{s[-tail:]}"


def draw_graph_from_arrays(
    *,
    # Edges (each position i describes the same edge across arrays)
    txt_edge_pairs: Sequence[Tuple[str, str]],   # [("txtsrc1","txtdest1"), ...]
    id_edge_pairs: Sequence[Tuple[int, int]],    # [(src_id, dst_id), ...]
    xpath_edge_pairs: Sequence[Tuple[str, str]], # [(src_xpath, dst_xpath), ...]
    probs: Sequence[float],                      # [p0, p1, ...]

    # Display/layout options
    title: str = "Graph",
    figsize: Tuple[int, int] = (12, 9),
    seed: int = 42,
    edge_cmap: str = "viridis",
    min_edge_width: float = 0.6,
    max_edge_width: float = 4.0,
    arrow_size: int = 12,
    k_layout: Optional[float] = None,   # spring_layout 'k'; None lets NX choose
    gravity: float = 0.0,               # small negative to push apart components
    show_colorbar: bool = True,
    directed: bool = True,
) -> Dict[str, Any]:
    """
    Build and display a graph image from parallel edge arrays.
    - Node labels = truncated text ('first10...last10') plus node id on a new line.
    - If the same (u,v) appears multiple times, the highest-probability edge is kept.

    Returns: {"G", "pos", "fig", "ax"}
    """
    # ---- Validate lengths ----
    n = len(probs)
    if not (len(txt_edge_pairs) == len(id_edge_pairs) == len(xpath_edge_pairs) == n):
        raise ValueError(
            f"Length mismatch: probs={n}, "
            f"txt_edge_pairs={len(txt_edge_pairs)}, "
            f"id_edge_pairs={len(id_edge_pairs)}, "
            f"xpath_edge_pairs={len(xpath_edge_pairs)}"
        )

    # ---- Build graph ----
    G_cls = nx.DiGraph if directed else nx.Graph
    G = G_cls()

    # Stable node labels from first occurrence
    seen_label_for: Dict[int, str] = {}

    # Keep single highest-probability edge per (u,v)
    best_edge_idx: Dict[Tuple[int, int], int] = {}

    for i, ((u, v), (utxt, vtxt), (uxp, vxp), p) in enumerate(
        zip(id_edge_pairs, txt_edge_pairs, xpath_edge_pairs, probs)
    ):
        u = int(u); v = int(v); p = float(p)

        if u not in seen_label_for:
            seen_label_for[u] = f"{_truncate(utxt)}\n(id {u})"
        if v not in seen_label_for:
            seen_label_for[v] = f"{_truncate(vtxt)}\n(id {v})"

        key = (u, v)
        if key not in best_edge_idx or float(probs[best_edge_idx[key]]) < p:
            best_edge_idx[key] = i

    # Add nodes with labels
    for node_id, label in seen_label_for.items():
        G.add_node(node_id, label=label)

    # Add edges (only the best per pair)
    for (u, v), i in best_edge_idx.items():
        s_txt, t_txt = txt_edge_pairs[i]
        s_xp, t_xp = xpath_edge_pairs[i]
        p = float(probs[i])
        G.add_edge(
            u, v,
            prob=p,
            src_text=s_txt, tgt_text=t_txt,
            src_xpath=s_xp, tgt_xpath=t_xp
        )

    # ---- Layout ----
    pos = nx.spring_layout(G, seed=seed, k=k_layout)
    if gravity != 0.0:
        for n, (x, y) in pos.items():
            r = (x**2 + y**2) ** 0.5 + 1e-12
            pos[n] = (x + gravity * x / r, y + gravity * y / r)

    # ---- Edge styling by probability ----
    edge_probs = np.array([G.edges[e]["prob"] for e in G.edges()], dtype=float)
    if len(edge_probs) == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = float(edge_probs.min()), float(edge_probs.max())
        if vmin == vmax:
            vmin, vmax = vmin - 1e-9, vmax + 1e-9

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = get_cmap(edge_cmap)
    edge_colors = [cmap(norm(G.edges[e]["prob"])) for e in G.edges()]
    widths = (min_edge_width + (max_edge_width - min_edge_width) * norm(edge_probs)) if len(edge_probs) else []

    # ---- Draw ----
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=900, linewidths=1.0, edgecolors="black")
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes()}, font_size=8, ax=ax)

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        arrows=directed,
        width=list(widths) if len(edge_probs) else 1.0,
        edge_color=edge_colors if len(edge_probs) else "k",
        arrowsize=arrow_size,
        connectionstyle="arc3,rad=0.12",
        min_source_margin=10, min_target_margin=10,
    )

    if G.number_of_edges() > 0:
        edge_labels = {(u, v): f"{d['prob']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    ax.axis("off")

    if show_colorbar and G.number_of_edges() > 0:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Edge probability")

    plt.tight_layout()
    return {"G": G, "pos": pos, "fig": fig, "ax": ax}