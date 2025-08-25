import math
from typing import Iterable, Optional, Sequence, Tuple, Dict, Any, Union

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap


def _truncate(text: str, head=10, tail=10) -> str:
    if text is None:
        return ""
    if len(text) <= head + tail + 3:
        return text
    return f"{text[:head]}...{text[-tail:]}"


def draw_graph_from_arrays(
    *,
    # Edges (each position i describes the same edge across arrays)
    txt_edge_pairs: Optional[Sequence[Tuple[str, str]]] = None,   # e.g. [["txtsrc1","txtdest1"], ...]
    id_edge_pairs: Optional[Union[np.ndarray, Sequence[Tuple[int, int]]]] = None,  # node id pairs (int)
    xpath_edge_pairs: Optional[Sequence[Tuple[str, str]]] = None,  # optional descriptor per node on the edge
    probs: Optional[Sequence[float]] = None,                       # edge probabilities (len == num_edges)

    # Node catalog (recommended): texts indexed by node id
    node_texts: Optional[Union[Sequence[str], np.ndarray]] = None, # e.g. txts = np.array(normtxt)

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

    You can supply edges by node IDs (recommended) and pass `node_texts` (1-D array)
    so labels come from that catalog. If you don't have `node_texts`, you can supply
    `txt_edge_pairs` and labels will be inferred from seen texts.

    Parameters
    ----------
    txt_edge_pairs : list/array of (str, str), optional
        Per-edge node text pairs. If `node_texts` is provided, this is optional.
    id_edge_pairs : list/array of (int, int), optional
        Per-edge node id pairs. Preferred if you also pass `node_texts`.
    xpath_edge_pairs : list/array of (str, str), optional
        Per-edge node xpath pairs (stored as edge attributes).
    probs : list/array of float
        Edge probabilities in [0, 1] (used for edge color/width).
    node_texts : 1-D list/array of str, optional
        Text for each node id (index = node id). Used to label nodes cleanly.
        If omitted, labels fall back to any text seen in `txt_edge_pairs`.
    Returns
    -------
    dict with keys:
        - "G": the NetworkX graph
        - "pos": node positions
        - "fig": matplotlib figure
        - "ax": matplotlib axes
    """

    # ---- Validate inputs ----
    if probs is None:
        raise ValueError("`probs` must be provided (1-D sequence of edge probabilities).")

    probs = np.asarray(probs).astype(float).tolist()

    # Basic length checks across parallel edge arrays
    num_edges = len(probs)

    def _length_or_none(x):
        return None if x is None else len(x)

    for name, arrlen in [
        ("txt_edge_pairs", _length_or_none(txt_edge_pairs)),
        ("id_edge_pairs", _length_or_none(id_edge_pairs)),
        ("xpath_edge_pairs", _length_or_none(xpath_edge_pairs)),
    ]:
        if arrlen is not None and arrlen != num_edges:
            raise ValueError(f"Length mismatch: {name} has {arrlen}, but probs has {num_edges}.")

    # ---- Build graph ----
    G_cls = nx.DiGraph if directed else nx.Graph
    G = G_cls()

    # If we have node_texts indexed by id, prepare a label lookup.
    node_texts_lookup: Dict[int, str] = {}
    if node_texts is not None:
        # Ensure we can index by int ids safely
        node_texts = np.asarray(node_texts, dtype=object)
        node_texts_lookup = {int(i): ("" if node_texts[i] is None else str(node_texts[i]))
                             for i in range(len(node_texts))}

    # If no id_edge_pairs provided, we’ll synthesize ids from seen texts (stable mapping)
    synth_id = False
    if id_edge_pairs is None:
        if txt_edge_pairs is None:
            raise ValueError("Provide at least one of `id_edge_pairs` or `txt_edge_pairs`.")
        synth_id = True
        id_map: Dict[str, int] = {}
        next_id = 0
        id_edge_pairs = []
        for s, t in txt_edge_pairs:
            for node_text in (s, t):
                if node_text not in id_map:
                    id_map[node_text] = next_id
                    next_id += 1
            id_edge_pairs.append((id_map[s], id_map[t]))
        # If no node_texts catalog was passed, derive it from seen text
        if not node_texts_lookup:
            inv = {v: k for k, v in id_map.items()}
            node_texts_lookup = {i: inv[i] for i in range(len(inv))}

    # Normalize id_edge_pairs to Python list of tuples
    id_edge_pairs = [tuple(map(int, e)) for e in id_edge_pairs]  # type: ignore

    # Add nodes with labels (truncate + id)
    def _node_label(node_id: int) -> str:
        base = node_texts_lookup.get(node_id, f"node_{node_id}")
        return f"{_truncate(str(base))}\n(id {node_id})"

    for u, v in id_edge_pairs:
        if u not in G:
            G.add_node(u, label=_node_label(u))
        if v not in G:
            G.add_node(v, label=_node_label(v))

    # Add edges with attributes
    # If multiple edges between same pair, keep the one with highest prob (cleaner drawing).
    best_edge: Dict[Tuple[int, int], Tuple[int, float]] = {}  # (u,v) -> (idx, prob)
    for i, (u, v) in enumerate(id_edge_pairs):
        p = float(probs[i])
        key = (u, v)
        if key not in best_edge or p > best_edge[key][1]:
            best_edge[key] = (i, p)

    for (u, v), (i, p) in best_edge.items():
        attrs = {"prob": p}
        if txt_edge_pairs is not None:
            s_txt, t_txt = txt_edge_pairs[i]
            attrs["src_text"] = s_txt
            attrs["tgt_text"] = t_txt
        if xpath_edge_pairs is not None:
            s_xp, t_xp = xpath_edge_pairs[i]
            attrs["src_xpath"] = s_xp
            attrs["tgt_xpath"] = t_xp
        G.add_edge(u, v, **attrs)

    # ---- Layout ----
    # Spring layout with a little "gravity" option (shift everything outward if negative).
    rng = np.random.default_rng(seed)
    pos = nx.spring_layout(G, seed=seed, k=k_layout)
    if gravity != 0.0:
        # push/pull nodes radially from origin
        for n, (x, y) in pos.items():
            r = math.hypot(x, y) + 1e-9
            pos[n] = (x + gravity * x / r, y + gravity * y / r)

    # ---- Styling: edge colors/widths by probability ----
    edge_probs = np.array([G.edges[e]["prob"] for e in G.edges()])
    # Normalize into [0,1] even if probs aren’t strictly within [0,1]
    norm = Normalize(vmin=float(np.min(edge_probs)) if len(edge_probs) else 0.0,
                     vmax=float(np.max(edge_probs)) if len(edge_probs) else 1.0)
    cmap = get_cmap(edge_cmap)
    edge_colors = [cmap(norm(G.edges[e]["prob"])) for e in G.edges()]

    # Width scale
    if len(edge_probs):
        widths = min_edge_width + (max_edge_width - min_edge_width) * norm(edge_probs)
    else:
        widths = []

    # ---- Draw ----
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)

    # Nodes
    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=900, linewidths=1.0, edgecolors="black"
    )

    # Node labels
    node_labels = {n: G.nodes[n]["label"] for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)

    # Edges
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,
        width=list(widths) if isinstance(widths, np.ndarray) else widths,
        edge_color=edge_colors,
        arrowsize=arrow_size,
        connectionstyle="arc3,rad=0.1",  # slight curvature to reduce overlap
        min_source_margin=10,
        min_target_margin=10,
    )

    # Edge labels (optional: show probs with 2 decimals). Comment out if too busy.
    edge_labels = {(u, v): f"{G.edges[(u, v)]['prob']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

    ax.axis("off")

    # Colorbar for probability
    if show_colorbar and len(edge_probs):
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Edge probability")

    plt.tight_layout()
    return {"G": G, "pos": pos, "fig": fig, "ax": ax}