from typing import Sequence, Tuple, Optional, Dict, Any, Literal
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap


def _truncate(text: str, head: int = 10, tail: int = 10) -> str:
    s = "" if text is None else str(text)
    if len(s) <= head + tail + 3:
        return s
    return f"{s[:head]}...{s[-tail:]}"

import math

def _spring_separate_components(G, *, seed=42, k=None, weight="prob", spacing=4.0, directed=True):
    # Components
    comps = list(nx.weakly_connected_components(G) if directed else nx.connected_components(G))
    cols = math.ceil(math.sqrt(len(comps)))  # grid width
    pos = {}

    for i, comp_nodes in enumerate(comps):
        sub = G.subgraph(comp_nodes)
        # pick a k appropriate to the subgraph size if not provided
        k_sub = k if k is not None else 3.0 / np.sqrt(max(sub.number_of_nodes(), 1))
        pos_sub = nx.spring_layout(sub, seed=seed, k=k_sub, weight=weight)

        # normalize sub-layout to ~unit box (so spacing works regardless of size)
        xs, ys = zip(*pos_sub.values())
        dx = (max(xs) - min(xs)) or 1.0
        dy = (max(ys) - min(ys)) or 1.0

        gx, gy = i % cols, i // cols  # grid cell
        ox, oy = gx * spacing, gy * spacing

        for n, (x, y) in pos_sub.items():
            pos[n] = (ox + (x - min(xs)) / dx, oy + (y - min(ys)) / dy)

    return pos


def _hierarchical_layout(
    G: nx.DiGraph,
    *,
    rankdir: Literal["TB", "BT", "LR", "RL"] = "LR",
    layer_spacing: float = 4.0,
    node_spacing: float = 2.0,
) -> Dict[Any, Tuple[float, float]]:
    """
    Compact hierarchical layout without external deps.
    Contracts SCCs -> DAG, assigns layers by longest-path depth, and spaces nodes.
    """
    # Contract strongly connected components to a DAG
    sccs = list(nx.strongly_connected_components(G))
    scc_id_of: Dict[Any, int] = {}
    for i, comp in enumerate(sccs):
        for n in comp:
            scc_id_of[n] = i

    D = nx.DiGraph()
    D.add_nodes_from(range(len(sccs)))
    for u, v in G.edges():
        su, sv = scc_id_of[u], scc_id_of[v]
        if su != sv:
            D.add_edge(su, sv)

    # Longest-path layering on the DAG
    order = list(nx.topological_sort(D)) if D.number_of_nodes() else []
    layer = {u: 0 for u in order}
    for u in order:
        for v in D.successors(u):
            layer[v] = max(layer[v], layer[u] + 1)

    # Map each original node to a layer
    node_layer: Dict[Any, int] = {}
    for comp_idx, comp in enumerate(sccs):
        L = layer.get(comp_idx, 0)
        for n in comp:
            node_layer[n] = L

    # Group nodes by layer
    layers: Dict[int, list] = {}
    for n, L in node_layer.items():
        layers.setdefault(L, []).append(n)

    # Stable order within each layer
    for L in layers:
        layers[L] = sorted(layers[L], key=lambda x: (G.in_degree(x), G.out_degree(x), x))

    # Coordinates
    pos: Dict[Any, Tuple[float, float]] = {}
    for L in sorted(layers):
        nodes = layers[L]
        count = len(nodes)
        offsets = [(i - (count - 1) / 2) * node_spacing for i in range(count)]

        if rankdir in ("TB", "BT"):
            y = (-L if rankdir == "TB" else L) * layer_spacing
            for xoff, n in zip(offsets, nodes):
                pos[n] = (xoff, y)
        else:  # LR or RL
            x = (L if rankdir == "LR" else -L) * layer_spacing
            for yoff, n in zip(offsets, nodes):
                pos[n] = (x, yoff)

    return pos


def draw_graph_from_arrays(
    *,
    # Edges (each position i describes the same edge across arrays)
    txt_edge_pairs: Sequence[Tuple[str, str]],   # [("txtsrc1","txtdest1"), ...]
    id_edge_pairs: Sequence[Tuple[int, int]],    # [(src_id, dst_id), ...]
    xpath_edge_pairs: Sequence[Tuple[str, str]], # [(src_xpath, dst_xpath), ...]
    probs: Sequence[float],                      # [p0, p1, ...]

    # Display/layout options
    title: str = "Graph",
    figsize: Tuple[int, int] = (14, 10),
    seed: int = 42,
    layout: Literal["hierarchical", "spring", "kk"] = "hierarchical",
    rankdir: Literal["TB", "BT", "LR", "RL"] = "LR",
    layer_spacing: float = 4.0,   # ↑ spread between layers
    node_spacing: float = 2.0,    # ↑ spread within a layer
    edge_cmap: str = "viridis",
    min_edge_width: float = 0.6,
    max_edge_width: float = 4.0,
    arrow_size: int = 12,
    k_layout: Optional[float] = None,   # spring_layout 'k'
    gravity: float = 0.0,               # small negative to push apart components (spring/kk)
    show_colorbar: bool = True,
    directed: bool = True,
    node_size: int = 900,
    font_size: int = 9,
) -> Dict[str, Any]:
    """
    Build & display a graph image from parallel edge arrays.
    - Node labels = truncated text ('first10...last10') + node id.
    - If (u,v) repeats, keeps the highest-probability edge.
    - Layouts: 'hierarchical' (default), 'spring', 'kk' (no external deps).
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

    seen_label_for: Dict[int, str] = {}
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

    for node_id, label in seen_label_for.items():
        G.add_node(node_id, label=label)

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

    # ---- Layouts (no external deps) ----
    if layout == "hierarchical":
        pos = _hierarchical_layout(G, rankdir=rankdir, layer_spacing=layer_spacing, node_spacing=node_spacing)
    elif layout == "kk":
        pos = nx.kamada_kawai_layout(G, weight="prob")
        if gravity != 0.0:
            for n, (x, y) in pos.items():
                r = (x**2 + y**2) ** 0.5 + 1e-12
                pos[n] = (x + gravity * x / r, y + gravity * y / r)
    else:  # "spring"
        if k_layout is None:
            k_layout = 3.0 / (np.sqrt(max(G.number_of_nodes(), 1)))
        # keep components separate:
        pos = _spring_separate_components(G, seed=seed, k=k_layout, weight="prob", spacing=layer_spacing, directed=directed)
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

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size, linewidths=1.0, edgecolors="black")
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["label"] for n in G.nodes()}, font_size=font_size, ax=ax, font_color="darkorange")

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        arrows=True,
        width=list(widths) if len(edge_probs) else 1.0,
        edge_color=edge_colors if len(edge_probs) else "k",
        arrowsize=arrow_size,
        connectionstyle="arc3,rad=0.0" if layout == "hierarchical" else "arc3,rad=0.12",
        min_source_margin=10, min_target_margin=10,
    )

    if G.number_of_edges() > 0:
        edge_labels = {(u, v): f"{d['prob']:.2f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=max(7, font_size-1), ax=ax)

    ax.axis("off")
    ax.margins(0.12)  # breathing room around the layout

    if show_colorbar and G.number_of_edges() > 0:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Edge probability")

    plt.tight_layout()
    plt.show()
    return {"G": G, "pos": pos, "fig": fig, "ax": ax}
