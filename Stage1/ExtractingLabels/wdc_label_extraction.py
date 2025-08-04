import sys
sys.path.insert(1, r"C:/Users/micha/Documents/Imperial Courses/Thesis/IAEA-thesis")
import json
from pathlib import Path
from typing import Dict
from lxml import etree
import html as htm
from Stage1.ExtractingGraphs.verifyGraphSize import verify_A_size, import_npz
from Stage1.ExtractingGraphs.HTMLtoGraph import EdgeFeatures
from Stage1.tree_helpers import *
from Stage1.ExtractingLabels.swde_label_extraction import *
import numpy as np
from scipy import sparse
import pandas as pd
import random

from collections import defaultdict
import itertools

def _get_pairs(tree):
    # first build your index mappings
    nodeToIndex, indexToNode = bfs_index_map(tree)

    # collect for each itemscope-parent the list of its itemprop/itemscope children
    toLink = defaultdict(list)   # parent_index -> [ child_index, ... ]

    for node, idx in nodeToIndex.items():
        # only consider nodes that carry itemprop or itemscope
        if 'itemprop' in node.attrib or 'itemscope' in node.attrib:
            # climb until we find the closest itemscope ancestor
            parent = node.getparent()
            while parent is not None and 'itemscope' not in parent.attrib:
                parent = parent.getparent()

            if parent is not None and get_node_text(node) != "":
                parent_idx = nodeToIndex[parent]
                toLink[parent_idx].append(idx)

    for parent in toLink:
        if get_direct_node_text(indexToNode[parent]):
            toLink[parent].append(parent)

    # now build all unordered pairs among the siblings of each parent
    pairs = []
    for child_indices in toLink.values():
        # need at least two children to make a pair
        if len(child_indices) >= 5:
            pairs.extend(itertools.combinations(child_indices, 2))

    return pairs

def withinSameitemscope(nodei,nodej,tree):
    parenti = nodei
    while parenti is not None and 'itemscope' not in parenti.attrib:
        parenti = parenti.getparent()

    parentj = nodej
    while parentj is not None and 'itemscope' not in parentj.attrib:
        parentj = parentj.getparent()

    return parenti is not None and parenti==parentj

def _createLabels(dataPath, tree: etree._ElementTree, coords: list[tuple[int, int]]) -> tuple[list, list]:
    """Outputs edge_index, edge_features"""
    #Get everything needed for EdgeFeatures
    index, nodes = bfs_index_map(tree)
    node_list = []
    for i, j in coords:
        node_list.append(nodes[i])
        node_list.append(nodes[j])

    # Need X, bbox, XPaths and parent/depth maps
    X = import_npz(dataPath / "X.npz")
    bbox = import_bbox(dataPath / "bbox.csv")
    XPaths: Dict[etree._Element, str] = {}
    for el in index:
        xp = tree.getpath(el)
        XPaths[el] = xp
    parentMap, depthMap = build_parent_and_depth_maps(tree)

    featurespositive = []
    labelpositive = []
    for i, j in coords:
        featurespositive.append(EdgeFeatures(
            edgeStart=i,
            edgeEnd=j,
            edgeStartNode=nodes[i],
            edgeEndNode=nodes[j],
            X=X,
            bboxs=bbox,
            parentMap=parentMap,
            depthMap=depthMap,
            XPaths=XPaths
        ))
        labelpositive.append(1)

    maxhop = np.max([row[-5] for row in featurespositive])

    featuresnegative = []
    labelnegative = []
    edgeIndexnegative = []
    node_list_copy = node_list.copy()
    candidateNodes = [i for i, _ in iter_elements_with_direct_text(tree)]

    sinceLastAppended = 0
    randomAdditions = 0
    while len(featuresnegative) < 2*len(featurespositive) and sinceLastAppended < 10000:
        sinceLastAppended+=1
        node = random.choice(node_list_copy)
        candidate = random.choice(candidateNodes)
        nhops = compute_hops(node, candidate, parent_map=parentMap, depth_map=depthMap)

        i, j = index[node], index[candidate]
        if connected(i, j, coords, index, nodes) or connected(j, i, coords, index, nodes) or withinSameitemscope(node,candidate,tree) or (i,j) in edgeIndexnegative or (j,i) in edgeIndexnegative:
            continue
        if nhops <= maxhop or randomAdditions < 0.02*len(featuresnegative):
            if nhops > maxhop: randomAdditions+=1 
            sinceLastAppended=0
            
            featuresnegative.append(EdgeFeatures(
                edgeStart=i,
                edgeEnd=j,
                edgeStartNode=node,
                edgeEndNode=candidate,
                X=X,
                bboxs=bbox,
                parentMap=parentMap,
                depthMap=depthMap,
                XPaths=XPaths
            ))
            labelnegative.append(0)
            edgeIndexnegative.append((i,j))

            featuresnegative.append(EdgeFeatures(
                edgeStart=j,
                edgeEnd=i,
                edgeStartNode=candidate,
                edgeEndNode=node,
                X=X,
                bboxs=bbox,
                parentMap=parentMap,
                depthMap=depthMap,
                XPaths=XPaths
            ))
            labelnegative.append(0)
            edgeIndexnegative.append((j,i))

            if np.random.random() < 0.5:
                node_list_copy.remove(node)
        
        
    
    label_index = np.concat((coords, edgeIndexnegative))
    label_features = np.concat((featurespositive, featuresnegative))
    label_value = np.concat((labelpositive, labelnegative))
    return label_index, label_features, label_value

def wdc_label_extraction(htmlFile: Path, dataPath:Path, save=False, verifyTreeAgainstFile=False, displayLabels=False, displaynegativeLabels=False) -> list[tuple]:

    tree = load_html_as_tree(htmlFile)
    treeSize = sum(1 for _ in tree.iter())
    if verifyTreeAgainstFile:
        verify_A_size(treeSize, dataPath  / "A.npz")
    
    coords = _get_pairs(tree)
    if len(coords) == 0:
        raise ValueError("nothing to save – no valid (int, int) pairs found")

    label_index, label_features, label_value = _createLabels(dataPath, tree, coords)

    if displayLabels:
        display_labels(tree, label_index[:len(coords)])
    if displaynegativeLabels:
        display_labels(tree, label_index[len(coords):])

    if save:
        save_coords_to_npz(label_index, label_features, label_value, dataPath)

    return label_index, label_features, label_value

if __name__ == "__main__":
    ANCHORHTML = Path("./data/wdc_microdata_html")
    ANCHORGRAPHS = Path("./data/wdc_microdata_HTMLgraphs")
    #TARGETFOLDER = Path("")

    htmlFolder = ANCHORHTML #/ TARGETFOLDER
    temp = list(ANCHORGRAPHS.rglob("A.npz"))
    html_files = [Path(str(x).replace("_HTMLgraphs", "_html")).parent.with_suffix(".html") for x in temp]
    rel   = html_files[0].relative_to(htmlFolder)
    dataPath = (ANCHORGRAPHS / rel).with_suffix("")

    wdc_label_extraction(html_files[0], dataPath, save=False, verifyTreeAgainstFile=True, displayLabels=True, displaynegativeLabels=True)