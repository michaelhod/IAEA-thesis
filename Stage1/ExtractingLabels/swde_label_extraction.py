import sys
sys.path.insert(1, r"C:/Users/micha/Documents/Imperial Courses/Thesis/IAEA-thesis")
import json
from pathlib import Path
from typing import Dict
from lxml import etree, html
from Stage1.ExtractingGraphs.verifyGraphSize import verify_A_size
from Stage1.tree_helpers import *
import numpy as np
from scipy import sparse

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)

def iterate_pairs(jsonFile, fileName: str):
    for labels, values in jsonFile[fileName].items():
        parts = [label.strip() for label in labels.split("|")]
        for i in range(len(parts) - 1):
            yield (parts[i], parts[i+1])
        for value in values:
            yield (parts[-1], value)

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
    needle_lower = needle.lower()

    # 1. Fast‑path: direct matches
    direct_matches = [el for el, part_text in iter_elements_with_direct_text(tree)
                      if needle_lower in part_text.lower()]  # type: ignore[arg-type]
    if direct_matches:
        return direct_matches

    # 2. Fallback: elements whose text is a *part* of the needle
    substr_candidates = [(el, part_text) for el, part_text in iter_elements_with_direct_text(tree)
                         if part_text and part_text.strip().lower() in needle_lower]
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
            txt = get_node_text(node) or ""
            if txt and needle_lower in txt.lower():
                break  # node now holds the ancestor with full match
            node = node.getparent()
        target = node
        if target and target not in seen:
            seen.add(target)                
            results.append(target)

            #Also add all parents to seen
            parent = target.getparent()
            while parent and parent not in seen:
                seen.add(parent)
                parent = parent.getparent()
                if parent in seen:
                    results.remove(parent)

    return results

def _find_exact_matches(potential_matches: list[etree._Element], needle:str) -> list:
    """
    Finds exact matches from nodes in potential_matches
    
    If no matches, returns the whole list"""
    exact_matches = []
    for node in potential_matches:
        txt = get_node_text(node).lower()

        if txt == needle.lower():
            exact_matches.append(node)

    return exact_matches if len(exact_matches) > 0 else potential_matches

def _closest_for_pair(tree: etree._ElementTree, left: str, right: str):
    """Compute closest (by line diff) element pair for *left*, *right*."""
    parent_map, depth_map = build_parent_and_depth_maps(tree)
    
    nodes_left = _find_matches(tree, left, depth_map)
    nodes_right = _find_matches(tree, right, depth_map)

    if not nodes_left or not nodes_right:
        # No match for one or both texts
        if not nodes_left:
            print(f"\"{left}\" not found in {left} | {right}")
        if not nodes_right:
            print(f"\"{right}\" not found in {left} | {right}")
        
        return f"{left} | {right}"

    #Find unique exact match, otherwise best guess
    nodes_left = _find_exact_matches(nodes_left, left)
    nodes_right = _find_exact_matches(nodes_right, right)

    bfs_indices, _ = bfs_index_map(tree)

    best_hops = float("inf")
    best_pair = (None, None)
    for a in nodes_left:
        for b in nodes_right:
            hops = compute_hops(a, b, parent_map, depth_map)
            if hops > 0 and hops < best_hops: #Can't be itself
                best_hops = hops
                best_pair = (a, b)

    return (bfs_indices[best_pair[0]], bfs_indices[best_pair[1]])

def label_extraction(htmlFile: Path, jsonContent, verifyTreeAgainstFile = None, displayLabels=False) -> None:

    tree = load_html_as_tree(htmlFile)

    if verifyTreeAgainstFile:
        verify_A_size(sum(1 for _ in tree.iter()), verifyTreeAgainstFile)
    
    htmlName = htmlFile.name

    results = [_closest_for_pair(tree, left, right) for left, right in iterate_pairs(jsonContent, htmlName)]

    if displayLabels:
        display_labels(tree, results)

    return results

def display_labels(tree, results):
    """input tree of html and results of pairs"""
    coords = [(coord[0], coord[1]) for coord in results if isinstance(coord[0], int) and isinstance(coord[1], int)]
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

def save_labels_to_npz(labels, graphSize, out_dir: Path):
    coords = [(coord[0], coord[1]) for coord in labels if isinstance(coord[0], int) and isinstance(coord[1], int)]
    if not coords:
        raise ValueError("nothing to save – no valid (int, int) pairs found")
    
    mask = np.zeros((graphSize, graphSize), dtype=np.uint8)
    for i, j in coords:
        mask[i, j] = 1

    mask = sparse.csr_matrix(mask)
    sparse.save_npz(out_dir / "labels.npz", mask, compressed=True)
    print(f"saved {out_dir}/labels.npz")

if __name__ == "__main__":
    ANCHORHTML = Path("./data/swde/sourceCode/sourceCode")
    ANCHORGRAPHS = Path("./data/swde_HTMLgraphs")
    TARGETFOLDER = Path("movie/movie/movie-allmovie(2000)")
    JSONFILE = "./data/swde_expanded_dataset/dataset/movie/movie-allmovie(2000).json"

    jsonContent = load_json(JSONFILE)

    htmlFolder = ANCHORHTML / TARGETFOLDER
    html_files = list(htmlFolder.rglob("*.htm"))
    htmlAPath = ANCHORGRAPHS / TARGETFOLDER / html_files[0].with_suffix("").name / "A.npz"

    label_extraction(html_files[0], jsonContent, verifyTreeAgainstFile=htmlAPath, displayLabels=True)

