import sys
sys.path.insert(1, r"C:/Users/micha/Documents/Imperial Courses/Thesis/IAEA-thesis")
import numpy as np
from seleniumFunctions import get_bbox, get_selenium_html, open_selenium
import json
from pathlib import Path
from safeHTMLTag import safe_name
from Stage1.tree_helpers import *

TAGSOFINTEREST = json.load(open("Stage1/ExtractingGraphs/tagsOfInterest.json", "r"))

def saveHTML(filepath, html):
    with filepath.open("w", encoding="utf-8") as f:
        f.write(html)

def EdgeFeatures(edgeStart, edgeEnd, edgeStartNode, edgeEndNode, X, bboxs, parentMap, depthMap, XPaths):
    features = [0]*(2*len(TAGSOFINTEREST)+7)

    # Copy all X features for each node
    for i in range(len(TAGSOFINTEREST)+1):
        features[i] = X[edgeStart, i]
    for i in range(len(TAGSOFINTEREST)+1):
        features[i+len(TAGSOFINTEREST)+1] = X[edgeEnd, i]
    # Num hops between nodes
    features[-5] = compute_hops(edgeStartNode, edgeEndNode, parentMap, depthMap)
    # Distances between nodes
    edgeEndXPath = XPaths[edgeEndNode]
    edgeStartXPath = XPaths[edgeStartNode]
    features[-4] = bboxs[edgeEndXPath]["x"]-bboxs[edgeStartXPath]["x"]
    features[-3] = bboxs[edgeEndXPath]["y"]-bboxs[edgeStartXPath]["y"]
    features[-2] = bboxs[edgeEndXPath]["width"]-bboxs[edgeStartXPath]["width"]
    features[-1] = bboxs[edgeEndXPath]["height"]-bboxs[edgeStartXPath]["height"]

    return features

def html_to_graph(filepath: Path, driver, OverwriteHTML=False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 1. Parse -----------------------------------------------------------------
    tree = load_html_as_tree(filepath)
    
    open_selenium(filepath, driver)
    if OverwriteHTML:
        html = get_selenium_html(driver=driver)
        tree = load_htmlstr_as_tree(html)

    # Flatten DOM into a list of element nodes (excluding NavigableStrings)
    nodes = bfs_index_map(tree)
    N = len(nodes)

    parentMap, depthMap = build_parent_and_depth_maps(tree)
    
    #Prime the XPATHS dict
    # 3. Build XPath map (lxml can do this natively)
    XPaths: Dict[etree._Element, str] = {}
    for el, i in nodes.items():
        xp = tree.getpath(el)
        XPaths[el] = xp
    
    # Collect bounding boxes
    bboxs = get_bbox(XPaths=list(XPaths.values()), driver=driver)

    if len(bboxs) != len(nodes):
        missing = []
        for path in XPaths.values():
            if path not in bboxs:
                missing.append(path)
        raise Exception(f"XPaths not found: {missing}")

    # 3. Build adjacency & graph ----------------------------------------------
    A = np.zeros((N, N), dtype=int)
    X = np.zeros((N, len(TAGSOFINTEREST) + 1), dtype=float)  # Node features
    edge_list = []
    edge_features = []

    # Populate Feature matrix, X
    for node, i in nodes.items():
        X[i, TAGSOFINTEREST[node.tag]] = 1 # One-hot tag
        if node.getparent() is not None and node.getparent() in nodes:
            siblings = [sib for sib in node.getparent() if sib in nodes]
            X[i, -1] = siblings.index(node) + 1 # Sibling index (1-based like XPath)
        else:
            X[i, -1] = 1

    # Populate adj matrix, A
    for node, edgeStart in nodes.items():
        # Connect to parent
        parent = node.getparent()
        if parent is not None and parent in nodes:
            edgeEnd = nodes[parent]
            A[edgeStart, edgeEnd] = 1
            edge_list.append((edgeStart, edgeEnd)) # Indexed as breadth first search
            edge_features.append(EdgeFeatures(edgeStart, edgeEnd, node, parent, X, bboxs, parentMap, depthMap, XPaths))


        # Connect to children
        for child in node:
            if child in nodes:
                edgeEnd = nodes[child]
                A[edgeStart, edgeEnd] = 1
                edge_list.append((edgeStart, edgeEnd)) # Indexed as breadth first search
                edge_features.append(EdgeFeatures(edgeStart, edgeEnd, node, child, X, bboxs, parentMap, depthMap, XPaths))

        # Connect siblings (same parent, direct siblings)
        siblings = [sib for sib in node.getparent() if sib in nodes] if node.getparent() is not None else []
        for sib in siblings:
            edgeEnd = nodes[sib]
            if sib != node:
                A[edgeStart, edgeEnd] = 1
                edge_list.append((edgeStart, edgeEnd)) # Indexed as breadth first search
                edge_features.append(EdgeFeatures(edgeStart, edgeEnd, node, sib, X, bboxs, parentMap, depthMap, XPaths))

    edge_index = np.array(edge_list)
    E = np.array(edge_features)

    if OverwriteHTML: #Replace the HTML with what selenium sees
        saveHTML(filepath, html)
        print(f"Overwrote {filepath}")

    return A, X, E, edge_index, bboxs

if __name__ == "__main__":
    from seleniumDriver import get_Driver, driver_init, restart_Driver
    html_file = Path("./data/swde/sourceCode/sourceCode/movie/movie/movie-allmovie(2000)/0000.htm")
    driver_init(True)
    restart_Driver(True)
    A, X, E, edge_index, bbox = html_to_graph(html_file, get_Driver())
    print("Adjacency Matrix:\n", A.shape)
    np.savetxt("X.csv", X, delimiter=",", fmt="%d")

    # np.savetxt("E1.csv", E[0,:,:], delimiter=",", fmt="%d")
    #print("Node Features:\n", X[49:])
    #print("Edge features: ", bbox)

    # Make a nx graph
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.from_numpy_array(A)
    position = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(12,8))
    nx.draw(G, position, ax, with_labels=True)
    plt.show()