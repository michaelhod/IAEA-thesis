import sys
sys.path.insert(1, r"C:\\Users\\micha\\Documents\\Imperial Courses\\Thesis\\IAEA-thesis")
import numpy as np
from Stage1.ExtractingGraphs.seleniumFunctions import get_bbox, get_selenium_html, open_selenium
import json
from pathlib import Path
from Stage1.ExtractingGraphs.safeHTMLTag import safe_name
from Stage1.tree_helpers import *
import torch
from torch_geometric.data import Data
import torch_geometric.transforms as T

TAGSOFINTEREST = json.load(open(r"C:\\Users\\micha\\Documents\\Imperial Courses\\Thesis\\IAEA-thesis\\Stage1\\ExtractingGraphs\\tagsOfInterest.json", "r"))

def saveHTML(filepath, html):
    with filepath.open("w", encoding="utf-8") as f:
        f.write(html)

def EdgeFeatures(edgeStart, edgeEnd, edgeStartNode, edgeEndNode, X, bboxs, parentMap, depthMap, XPaths):
    features = [0]*(2*len(TAGSOFINTEREST)+10)

    # Copy all X features for each node
    for i in range(len(TAGSOFINTEREST)+1):
        features[i] = X[edgeStart, i]
    for i in range(len(TAGSOFINTEREST)+1):
        features[i+len(TAGSOFINTEREST)+1] = X[edgeEnd, i]
    # Num hops between nodes
    features[-8] = np.log1p(compute_hops(edgeStartNode, edgeEndNode, parentMap, depthMap))
    
    # Distances between nodes
    edgeEndXPath = XPaths[edgeEndNode]
    edgeStartXPath = XPaths[edgeStartNode]
    vH = bboxs["/html"]["height"]+1e-9 #For generalisability
    vW = bboxs["/html"]["width"]+1e-9
    centrexStart = bboxs[edgeStartXPath]["x"] + bboxs[edgeStartXPath]["width"]/2.0
    centrexEnd = bboxs[edgeEndXPath]["x"] + bboxs[edgeEndXPath]["width"]/2.0
    centreyStart = bboxs[edgeStartXPath]["y"] + bboxs[edgeStartXPath]["height"]/2.0
    centreyEnd = bboxs[edgeEndXPath]["y"] + bboxs[edgeEndXPath]["height"]/2.0
    
    features[-7] = (centrexEnd-centrexStart)/vW
    features[-6] = (centreyEnd-centreyStart)/vH
    features[-5] = np.arctan2(centreyEnd-centreyStart, centrexEnd-centrexStart)
    features[-4] = np.log(np.clip(bboxs[edgeEndXPath]["width"]/(1e-9+bboxs[edgeStartXPath]["width"]), 1e-3, 1e3)) # Normalisation
    features[-3] = np.log(np.clip(bboxs[edgeEndXPath]["height"]/(1e-9+bboxs[edgeStartXPath]["height"]), 1e-3, 1e3))
    features[-2] = (bboxs[edgeEndXPath]["width"]-bboxs[edgeStartXPath]["width"])/vW
    features[-1] = (bboxs[edgeEndXPath]["height"]-bboxs[edgeStartXPath]["height"])/vH

    return features

def add_positional_encodings(X_np, edge_index_np, k_lap=8, rw_len=6):
    """
    X_np:          (N, D) numpy.float32
    edge_index_np: (E, 2) numpy.int64  (list of (u,v) pairs)
    returns:       X_with_pe as numpy (N, D + pe_dim)
    """
    # 1) to torch (CPU)
    x = torch.from_numpy(X_np).float()
    edge_index = torch.from_numpy(edge_index_np).long().t().contiguous()  # -> [2, E]

    # 2) add PEs (concats to x when attr_name=None)
    data = Data(x=x, edge_index=edge_index, num_nodes=x.size(0))
    if k_lap > 0:
        data = T.AddLaplacianEigenvectorPE(k=k_lap, attr_name=None)(data)
    data = T.AddRandomWalkPE(walk_length=rw_len, attr_name=None)(data)

    # 3) back to numpy
    return data.x.cpu().numpy()

def add_depth_deg(X, depthMap, edge_index, N, nodes):
    # 2) Depth per node (float)
    depth = np.zeros(N, dtype=np.float32)
    for el, i in nodes.items():
        depth[i] = float(depthMap[el])
    depth = np.log1p(depth) #scale-free, therefore hopefully generalize better

    # 3) In/out degree from directed edges you created
    if edge_index.size:
        out_deg = np.bincount(edge_index[:, 0], minlength=N).astype(np.float32)
        in_deg  = np.bincount(edge_index[:, 1], minlength=N).astype(np.float32)
    else:
        out_deg = np.zeros(N, dtype=np.float32)
        in_deg  = np.zeros(N, dtype=np.float32)
    in_deg, out_deg = np.log1p(in_deg), np.log1p(out_deg) #scale-free, therefore hopefully generalize better

    # 4) is_leaf flag from DOM structure (no element-children present in `nodes`)
    is_leaf = np.zeros(N, dtype=np.float32)
    for el, i in nodes.items():
        child_has_index = any((ch in nodes) for ch in el)  # only count elements present in `nodes`
        is_leaf[i] = 0.0 if child_has_index else 1.0

    # 5) Concatenate to X (adds 4 columns: depth, in_deg, out_deg, is_leaf)
    return np.concatenate(
        [X, np.column_stack([depth, in_deg, out_deg, is_leaf]).astype(np.float32)],
        axis=1
    )

def html_to_graph(filepath: Path, driver, OverwriteHTML=False, urlToOpen=None):
    # 1. Parse -----------------------------------------------------------------
    urlToOpen = Path(filepath).resolve().as_uri() if not urlToOpen else urlToOpen
    
    tree = load_html_as_tree(filepath)
    
    open_selenium(urlToOpen, driver)
    if OverwriteHTML:
        html = get_selenium_html(driver=driver)
        tree = load_htmlstr_as_tree(html)

    # Flatten DOM into a list of element nodes (excluding NavigableStrings)
    nodes, _ = bfs_index_map(tree)
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

    #Some final additions to X
    X = add_positional_encodings(X, edge_index, k_lap=8, rw_len=6)
    X = add_depth_deg(X, depthMap, edge_index, N, nodes)

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
    #np.savetxt("X.csv", X, delimiter=",", fmt="%d")

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