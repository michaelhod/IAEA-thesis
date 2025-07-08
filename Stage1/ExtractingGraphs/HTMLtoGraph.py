import numpy as np
import networkx as nx
from bboxPos import get_bbox
import json
from collections import defaultdict
from bs4 import BeautifulSoup, Tag

TAGSTOIGNORE = ["script", "style", "meta", "link", "noscript", "iframe", "svg", "canvas", "object", "embed"]
ALLTAGS = json.load(open("Stage1/ExtractingGraphs/allTags.json", "r"))
XPATHS = {} # This will be filled by the xpath function as we parse the HTML

def xpath(tag) -> str:
    """
    Build an XPath that uniquely identifies *bs4_element*
    inside the parsed document.

    • Tags are written literally: /html/body/div
    • Siblings with the same tag get a 1-based index: /div[3]
    """
    parts = []
    el = tag
    while el and el.name:                 # stop at the BeautifulSoup object
        if XPATHS.get(el) is not None:
            # If this element already has an XPath, return it
            parts.append(XPATHS[el][1:]) # Skip the first "/" as we will add it later
            break

        parent = el.parent
        # How many direct siblings share this tag?
        if parent:
            same_tag_sibs = [sib for sib in parent.find_all(el.name, recursive=False)]

            if len(same_tag_sibs) == 1:
                # unique → no index
                parts.append(el.name)
            else:
                # add 1-based position
                idx = same_tag_sibs.index(el) + 1
                parts.append(f"{el.name}[{idx}]")

        el = parent

    parts.reverse()
    XPATHS[tag] = "/" + "/".join(parts)  # Store the XPath for this tag
    return XPATHS[tag]  

def html_to_graph(html: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    html
        Raw HTML string

    Returns
    -------
    A : (N, N) np.ndarray[bool]
        Adjacency (parents <-> children and siblings).
    X : (N, F_node) np.ndarray[float]
        Node features: one-hot tag | sibling index.
    E : (N, N, F_edge) np.ndarray[float]
        Edge features: [X_i , X_j , hop_dist , Δx , Δy , Δh , Δw]
    """
    # 1. Parse -----------------------------------------------------------------
    soup = BeautifulSoup(html, "lxml")

    # Flatten DOM into a list of element nodes (excluding NavigableStrings)
    nodes: dict[Tag, int] = {} # list every node and index it for the adj matrix
    idx = 0
    for el in soup.descendants:
        if isinstance(el, Tag):
            nodes[el] = idx
            idx += 1
    N = len(nodes)

    #Prime the XPATHS dict
    for node in nodes:
        xpath(node)
    
    # Collect bounding boxes
    bboxs = get_bbox(html, XPaths=list(XPATHS.values()))

    # sibling indices
    # sibling_idx: list[int] = [] #What number sibling is each node (indexed the same as nodes)
    # for n in nodes:
    #     siblings = [sib for sib in n.parent.children if isinstance(sib, Tag)] if n.parent else []
    #     sibling_idx.append(siblings.index(n) if siblings else 0)

    # text frequencies
    # if text_freq_lookup is None:
    #     text_freq_lookup = defaultdict(float) # Dict of text: freq
    # text_freq = [text_freq_lookup.get(n.get_text(strip=True), 0.0) for n in nodes] # All index where many children down, the text is exactly the same

    # 3. Build adjacency & graph ----------------------------------------------
    A = np.zeros((N, N), dtype=int)
    X = np.zeros((N, 3), dtype=float)  # Node features
    E = np.zeros((N, N, 4), dtype=float)  # Edge features

    for node in nodes:
        # Connect to parent
        parent = node.parent
        if parent:
            A[nodes[node], nodes[parent]] = 1

            # add edge feature for parent-child
            

        # Connect to children
        for child in node.children:
            A[nodes[node], nodes[child]] = 1

        # Connect siblings (same parent, direct siblings)
        siblings = [sib for sib in node.parent.children if isinstance(sib, Tag)] if node.parent else []
        index = 1
        for i, sib in enumerate(siblings):
            if sib != node:
                A[nodes[node], nodes[sib]] = 1
            else:
                index = i + 1

        # Add to features
        X[nodes[node], 0] = ALLTAGS[node.name]  # One-hot tag
        X[nodes[node], 1] = index # Sibling index (1-based like XPath)



    return A, X, E
    G = nx.Graph()

    for idx, n in enumerate(nodes):
        G.add_node(idx)

    # parent/child edges
    for child_idx, child in enumerate(nodes):
        parent = child.parent
        if isinstance(parent, Tag):
            parent_idx = nodes.index(parent)
            G.add_edge(parent_idx, child_idx, hop=1)

    # sibling edges (same parent, direct siblings)
    for n in soup.find_all(True):
        siblings = [sib for sib in n.children if isinstance(sib, Tag)]
        for i, sib_i in enumerate(siblings):
            idx_i = nodes.index(sib_i)
            for sib_j in siblings[i + 1 :]:
                idx_j = nodes.index(sib_j)
                G.add_edge(idx_i, idx_j, hop=2)  # shortest path length between siblings

    A = nx.to_numpy_array(G, dtype=bool)

    # 4. Assemble node-feature matrix X ---------------------------------------
    # one-hot tag
    tag_onehot = np.zeros((N, len(ALLTAGS)), dtype=float)
    for i, n in enumerate(nodes):
        tag_onehot[i, ALLTAGS[n.name]] = 1.0

    sibling_idx_arr = np.array(sibling_idx, dtype=float).reshape(N, 1)
    text_freq_arr = np.array(text_freq, dtype=float).reshape(N, 1)

    X = np.hstack([tag_onehot, sibling_idx_arr, text_freq_arr])  # (N, F_node)
    F_node = X.shape[1]

    # 5. Edge-feature tensor ---------------------------------------------------
    F_edge = 2 * F_node + 5
    E = np.zeros((N, N, F_edge), dtype=float)
    # Pre-compute all-pairs shortest-path hop counts (small graph, so O(N^3) ok)
    hop_dists = dict(nx.all_pairs_shortest_path_length(G))

    for i, j in G.edges():
        # node features of K and V
        feat_i, feat_j = X[i], X[j]
        hop = hop_dists[i][j]

        # Δ coords
        dx = coords[j, 0] - coords[i, 0]
        dy = coords[j, 1] - coords[i, 1]
        dw = coords[j, 2] - coords[i, 2]
        dh = coords[j, 3] - coords[i, 3]

        edge_feat = np.concatenate([feat_i, feat_j, [hop, dx, dy, dh, dw]])
        E[i, j] = edge_feat
        E[j, i] = edge_feat  # symmetric

    return A, X, E

html_content = ""
with open("./Stage1/test.html", "r", encoding="utf-8") as f:
    html_content = f.read()
output = html_to_graph(html_content)
# print("Adjacency Matrix:\n", output[0])
# print("Node Features:\n", output[1])
# print("Edge features: ", output[2])