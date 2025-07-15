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

def EdgeFeatures(edgeStart, edgeEnd, X, bboxs, A=None, hops=None):
    features = [0]*8
    features[0] = X[edgeStart, 0]
    features[1] = X[edgeStart, 1]
    features[2] = X[edgeEnd, 0]
    features[3] = X[edgeEnd, 1]
    if hops:
        features[4] = hops
    else:
        raise Exception("NOT YET IMPLEMENTED THE HOPS CALCULATION BETWEEN TWO NODES")
    features[5] = bboxs[edgeEnd]["x"]-bboxs[edgeStart]["x"]
    features[6] = bboxs[edgeEnd]["y"]-bboxs[edgeStart]["y"]
    features[7] = bboxs[edgeEnd]["width"]-bboxs[edgeStart]["width"]
    features[8] = bboxs[edgeEnd]["height"]-bboxs[edgeStart]["height"]

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
    X = np.zeros((N, 2), dtype=float)  # Node features
    E = np.zeros((N, N, 8), dtype=float)  # Edge features

    # Populate Feature matrix, X
    for node in nodes:
        oneHot = [0]*len(ALLTAGS)
        oneHot[ALLTAGS[node.name]] = 1
        X[nodes[node], 0] = oneHot  # One-hot tag
        X[nodes[node], 1] = index # Sibling index (1-based like XPath)

    # Populate adj matrix, A
    for node in nodes:
        edgeStart = nodes[node]

        # Connect to parent
        parent = node.parent
        if parent:
            edgeEnd = nodes[parent]
            A[edgeStart, edgeEnd] = 1
            E[edgeStart, edgeEnd] = EdgeFeatures(edgeStart, edgeEnd, X, bboxs, hops=1)

        # Connect to children
        for child in node.children:
            edgeEnd = nodes[child]
            A[edgeStart, edgeEnd] = 1
            E[edgeStart, edgeEnd] = EdgeFeatures(edgeStart, edgeEnd, X, bboxs, hops=1)

        # Connect siblings (same parent, direct siblings)
        siblings = [sib for sib in node.parent.children if isinstance(sib, Tag)] if node.parent else []
        index = 1
        for i, sib in enumerate(siblings):
            edgeEnd = nodes[sib]
            if sib != node:
                A[edgeStart, edgeEnd] = 1
                E[edgeStart, edgeEnd] = EdgeFeatures(edgeStart, edgeEnd, X, bboxs, hops=1)
            else:
                index = i + 1

    return A, X, E

html_content = ""
with open("./Stage1/test.html", "r", encoding="utf-8") as f:
    html_content = f.read()
output = html_to_graph(html_content)
# print("Adjacency Matrix:\n", output[0])
# print("Node Features:\n", output[1])
# print("Edge features: ", output[2])