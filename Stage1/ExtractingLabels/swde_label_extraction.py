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
import numpy as np
from scipy import sparse
import pandas as pd
import random

def load_json_of_swde_file(htmlFilepath: str):
    htmlFilepath = htmlFilepath.split("/") if "/" in htmlFilepath else htmlFilepath.split("\\")
    jsonFilepath = f"./data/swde_expanded_dataset/dataset/{htmlFilepath[-3]}/{htmlFilepath[-2]}.json"
    with open(jsonFilepath, "r", encoding="utf-8") as fp:
        return json.load(fp)

def iterate_pairs(jsonFile, fileName: str):
    for labels, values in jsonFile[fileName].items():
        parts = [label.strip() for label in labels.split("|")]
        for part in parts.copy(): # This is to remove the specific case in metacritic where &&& does not refer to anything in the HTML
            if "&&&" in part: 
                parts.remove(part)
                #print(f"WARNING, {value} removed from json labels")
            elif part == "":
                parts.remove(part)
        for i in range(len(parts) - 1):
            yield (parts[i], parts[i+1])
        for value in values:
            if value == "":
                continue
            yield (parts[-1], value)

def _import_bbox(csv_path):
    df = pd.read_csv(csv_path, index_col=0)
    return df.to_dict(orient="index")

# Logic --------------------------------------------------------------

def _find_matches(tree: etree._ElementTree, needle: str, depth_map):
    """Return elements relevant to *needle*.

    Workflow
    --------
    1. **Fast‑path** – direct‑text match: any element whose *own* text contains
       the full ``needle`` is returned immediately.
    2. **Fallback** – substring ascent:
       • Collect elements whose text is a substring of the needle.
       • Sort them by descending text length (longest pieces first).
       • For each, walk *up* the DOM until we find an ancestor whose direct
         text contains the full needle; use that ancestor as the canonical
         match (or the original node if none found).
       • Deduplicate while preserving order.

    The procedure touches each node at most once per ascent chain, giving good
    performance even for large documents.
    """
    needle_lower = htm.unescape(needle).lower().replace("\r", "\n") 

    # 1. Fast‑path: direct matches
    direct_matches = [el for el, part_text in iter_elements_with_direct_text(tree)
                      if normalise_text(needle_lower) in normalise_text(part_text)] #Note, can have duplicates
    if direct_matches:
        return direct_matches

    # 2. Fallback: elements whose text is a *part* of the needle
    substr_candidates = [(el, part_text) for el, part_text in iter_elements_with_direct_text(tree)
                         if part_text and normalise_text(part_text) in normalise_text(needle_lower)] #Note, can have duplicates
    if not substr_candidates:
        return []  # nothing at all

    substr_candidates.sort(key=lambda el: depth_map.get(el, 0), reverse=True)

    results = []
    seen = set()

    for el, _ in substr_candidates:
        if el in seen:
            continue
        node = el
        # Walk up until an ancestor directly contains the full needle
        while node is not None:
            txt_normalised = get_node_text(node, True) or ""
            needle_normalised = normalise_text(needle_lower)

            if needle_normalised in txt_normalised: #Remove all non letters
                break  # node now holds the ancestor with full match
            node = node.getparent()
        
        target = node
        if target is not None and target not in seen:
            seen.add(target)                
            results.append(target)

            #Also add all parents to seen
            parent = target.getparent()
            while parent is not None and parent not in seen:
                seen.add(parent)
                parent = parent.getparent()
                if parent in seen and parent in results:
                    results.remove(parent)

    return results

def _find_exact_matches(potential_matches: list[etree._Element], needle:str) -> list:
    """
    Finds exact matches from nodes in potential_matches
    
    If no matches, returns the whole list"""
    exact_matches = []
    for node in potential_matches:
        txt = get_node_text(node, True)
        needle_normalised = normalise_text(needle)

        if txt == needle_normalised:
            exact_matches.append(node)

    return exact_matches if len(exact_matches) > 0 else potential_matches

def _closest_for_pair(tree: etree._ElementTree, left: str, right: str):
    """Compute closest (by line diff) element pair for *left*, *right*."""
    parent_map, depth_map = build_parent_and_depth_maps(tree)
    
    nodes_left = _find_matches(tree, left, depth_map) #Note, can have duplicates
    nodes_right = _find_matches(tree, right, depth_map) #Note, can have duplicates

    if not nodes_left or not nodes_right:
        # No match for one or both texts
        if not nodes_left:
            if "topic_entity_name" in left: #Ignore this particular case
                return None
            return f"Left not found in: {left} | {right}"
        else:
            return f"Right not found in: {left} | {right}"

    #Find unique exact match, otherwise best guess
    nodes_left = _find_exact_matches(nodes_left, left)
    nodes_right = _find_exact_matches(nodes_right, right)

    bfs_indices, _ = bfs_index_map(tree)

    best_hops = float("inf")
    best_pair = (None, None)
    for a in nodes_left:
        for b in nodes_right:
            hops = compute_hops(a, b, parent_map, depth_map)
            if hops < best_hops:
                best_hops = hops
                best_pair = (a, b)

                if hops == 0: #Can't be itself
                    return f"found the same tag for: {left} | {right}"

    return (bfs_indices[best_pair[0]], bfs_indices[best_pair[1]])

def connectparents(tree, i, j):
    nodeToindex, indexTonode = bfs_index_map(tree)
    nodeiText, nodejText = get_node_text(indexTonode[i]), get_node_text(indexTonode[j])

    parentsi = []
    parenti = indexTonode[i]
    while parenti is not None and get_node_text(parenti) == nodeiText:
        parentsi.append(nodeToindex[parenti])
        parenti = parenti.getparent()

    parentsj = []
    parentj = indexTonode[j]
    while parentj is not None and get_node_text(parentj) == nodejText:
        parentsj.append(nodeToindex[parentj])
        parentj = parentj.getparent()

    coords = []
    for pi in parentsi:
        for pj in parentsj:
            coords.append((pi,pj))

    return coords

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
    bbox = _import_bbox(dataPath / "bbox.csv")
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
    while len(featuresnegative) < len(featurespositive) and sinceLastAppended < 10000:
        sinceLastAppended+=1
        node = random.choice(node_list_copy)
        candidate = random.choice(candidateNodes)
        nhops = compute_hops(node, candidate, parent_map=parentMap, depth_map=depthMap)

        i, j = index[node], index[candidate]
        if (i, j) in coords or (j, i) in coords or (i,j) in edgeIndexnegative or (j,i) in edgeIndexnegative:
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

def _display_labels(tree, coords):
    """input tree of html and results of pairs"""
    _, nodes = bfs_index_map(tree)
    for i,j in coords:
        tag1 = nodes[i].tag
        txt1 = ''.join(nodes[i].itertext()).strip()
        #line1 = nodes[i].sourceline
        tag2 = nodes[j].tag
        txt2 = ''.join(nodes[j].itertext()).strip()
        #line2 = nodes[j].sourceline

        print(f"<{tag1}>{txt1}</{tag1}> -> <{tag2}>{txt2}</{tag2}>")#: \t\tSourceLine {line1} -> {line2}")
    print(len(coords))

def label_extraction(htmlFile: Path, jsonContent, dataPath:Path, save=False, verifyTreeAgainstFile=False, displayLabels=False, displaynegativeLabels=False) -> list[tuple]:

    tree = load_html_as_tree(htmlFile)
    treeSize = sum(1 for _ in tree.iter())
    if verifyTreeAgainstFile:
        verify_A_size(treeSize, dataPath  / "A.npz")
    
    htmlName = htmlFile.name
    results = [_closest_for_pair(tree, left, right) for left, right in iterate_pairs(jsonContent, htmlName)]
    tempcoords = [pair for coord in results if coord and isinstance(coord[0], int) and isinstance(coord[1], int) for pair in ((coord[0], coord[1]),(coord[1], coord[0]))]
    coords = []
    for pair in tempcoords:
        coords = coords + connectparents(tree, pair[0], pair[1])
    if len(coords) == 0:
        raise ValueError("nothing to save – no valid (int, int) pairs found")

    label_index, label_features, label_value = _createLabels(dataPath, tree, coords)

    if displayLabels:
        _display_labels(tree, label_index[:len(coords)])
    if displaynegativeLabels:
        _display_labels(tree, label_index[len(coords):])

    if save:
        _save_coords_to_npz(label_index, label_features, label_value, dataPath)

    return results, label_index, label_features, label_value

def _save_coords_to_npz(label_index, label_features, label_value, dataPath: Path):
    if len(label_index) == 0:
        raise ValueError("nothing to save – no valid (int, int) pairs found")
    
    label_features = sparse.csr_matrix(label_features)
    sparse.save_npz(dataPath / "labels.npz", label_features, compressed=True)
    np.save(dataPath / "label_index.npy", label_index)
    np.save(dataPath / "label_value.npy", label_value)

if __name__ == "__main__":
    ANCHORHTML = Path("./data/swde/sourceCode/sourceCode")
    ANCHORGRAPHS = Path("./data/swde_HTMLgraphs")
    TARGETFOLDER = Path("movie/movie/movie-allmovie(2000)")
    JSONFILE = "./data/swde_expanded_dataset/dataset/movie/movie/movie-allmovie(2000).json"

    htmlFolder = ANCHORHTML / TARGETFOLDER
    html_files = list(htmlFolder.rglob("*.htm"))
    dataPath = ANCHORGRAPHS / TARGETFOLDER / html_files[0].with_suffix("").name

    jsonContent = load_json_of_swde_file(str(html_files[0]))

    label_extraction(html_files[0], jsonContent, dataPath, save=False, verifyTreeAgainstFile=True, displayLabels=True, displaynegativeLabels=False)

