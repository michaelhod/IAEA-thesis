import numpy as np
from bboxPos import get_bbox
import json
from collections import defaultdict
from bs4 import BeautifulSoup, Tag

TAGSOFINTEREST = json.load(open("Stage1/ExtractingGraphs/tagsOfInterest.json", "r"))
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

def EdgeFeatures(edgeStart, edgeEnd, edgeStartXPath, edgeEndXPath, X, bboxs, A=None, hops=None):
    features = [0]*(2*len(TAGSOFINTEREST)+7)

    # Copy all X features for each node
    for i in range(len(TAGSOFINTEREST)+1):
        features[i] = X[edgeStart, i]
    for i in range(len(TAGSOFINTEREST)+1):
        features[i+len(TAGSOFINTEREST)+1] = X[edgeEnd, i]
    # Num hops between nodes
    if hops:
        features[-5] = hops
    else:
        raise Exception("NOT YET IMPLEMENTED THE HOPS CALCULATION BETWEEN TWO NODES")
    # Distances between nodes
    features[-4] = bboxs[edgeEndXPath]["x"]-bboxs[edgeStartXPath]["x"]
    features[-3] = bboxs[edgeEndXPath]["y"]-bboxs[edgeStartXPath]["y"]
    features[-2] = bboxs[edgeEndXPath]["width"]-bboxs[edgeStartXPath]["width"]
    features[-1] = bboxs[edgeEndXPath]["height"]-bboxs[edgeStartXPath]["height"]

    return features

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
        if isinstance(el, Tag) and el.name in TAGSOFINTEREST:
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
    X = np.zeros((N, len(TAGSOFINTEREST) + 1), dtype=float)  # Node features
    E = np.zeros((N, N, 2*len(TAGSOFINTEREST) + 7), dtype=float)  # Edge features

    # Populate Feature matrix, X
    for node in nodes:
        X[nodes[node], TAGSOFINTEREST[node.name]] = 1 # One-hot tag
        if node.parent and node.parent in nodes:
            siblings = [sib for sib in node.parent.children if isinstance(sib, Tag) and sib in nodes]
            X[nodes[node], -1] = siblings.index(node) + 1 # Sibling index (1-based like XPath)
        else:
            X[nodes[node], -1] = 1

    # Populate adj matrix, A
    for node in nodes:
        edgeStart = nodes[node]

        # Connect to parent
        parent = node.parent
        if parent and parent in nodes:
            edgeEnd = nodes[parent]
            A[edgeStart, edgeEnd] = 1
            E[edgeStart, edgeEnd] = EdgeFeatures(edgeStart, edgeEnd, XPATHS[node], XPATHS[parent], X, bboxs, hops=1)

        # Connect to children
        for child in node.children:
            if isinstance(child, Tag) and child in nodes:
                edgeEnd = nodes[child]
                A[edgeStart, edgeEnd] = 1
                E[edgeStart, edgeEnd] = EdgeFeatures(edgeStart, edgeEnd, XPATHS[node], XPATHS[child], X, bboxs, hops=1)

        # Connect siblings (same parent, direct siblings)
        siblings = [sib for sib in node.parent.children if isinstance(sib, Tag) and sib in nodes] if node.parent else []
        for i, sib in enumerate(siblings):
            edgeEnd = nodes[sib]
            if sib != node:
                A[edgeStart, edgeEnd] = 1
                E[edgeStart, edgeEnd] = EdgeFeatures(edgeStart, edgeEnd, XPATHS[node], XPATHS[sib], X, bboxs, hops=1)

    return A, X, E

# html_content = ""
# with open("./Stage1/test.html", "r", encoding="utf-8") as f:
#     html_content = f.read()
# A, X, E = html_to_graph(html_content)
# print("Adjacency Matrix:\n", A.shape)
# # np.savetxt("X.csv", X[49:], delimiter=",", fmt="%d")

# # np.savetxt("E1.csv", E[0,:,:], delimiter=",", fmt="%d")
# print("Node Features:\n", X[49:])
# print("Edge features: ", E.shape)

# # Make a nx graph
# import networkx as nx
# import matplotlib.pyplot as plt
# G = nx.from_numpy_array(A)
# position = nx.spring_layout(G)
# fig, ax = plt.subplots(figsize=(12,8))
# nx.draw(G, position, ax, with_labels=True)
# plt.show()