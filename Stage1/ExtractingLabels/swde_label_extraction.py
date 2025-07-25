import sys
sys.path.insert(1, r"C:/Users/micha/Documents/Imperial Courses/Thesis/IAEA-thesis")
import json
from pathlib import Path
from typing import Dict
from lxml import etree, html
from Stage1.ExtractingGraphs.verifyGraphSize import _verify_A_size
from Stage1.tree_helpers import *

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
            txt = ''.join(node.itertext()).strip() or ""
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

def _closest_for_pair(tree: etree._ElementTree, left: str, right: str):
    """Compute closest (by line diff) element pair for *left*, *right*."""
    parent_map, depth_map = build_parent_and_depth_maps(tree)
    
    nodes_left = _find_matches(tree, left, depth_map)
    nodes_right = _find_matches(tree, right, depth_map)

    if not nodes_left or not nodes_right:
        # No match for one or both texts
        return f"{left} | {right}"

    bfs_indices = bfs_index_map(tree)

    best_hops = float("inf")
    best_pair = (None, None)
    for a in nodes_left:
        for b in nodes_right:
            hops = compute_hops(a, b, parent_map, depth_map)
            if hops < best_hops:
                best_hops = hops
                best_pair = (a, b)

    return (bfs_indices[best_pair[0]], bfs_indices[best_pair[1]])

def label_extraction(htmlFile: Path, jsonContent, htmlFileApath) -> None:

    tree = load_html_as_tree(htmlFile)
    save_tree_html(tree, "./debug.htm")
    if not _verify_A_size(sum(1 for _ in tree.iter()), htmlFileApath):
        raise Exception("The length of the tree does not match the length of the Adj matrix")
    htmlName = htmlFile.name

    results = [_closest_for_pair(tree, left, right) for left, right in iterate_pairs(jsonContent, htmlName)]

    return results

def labels_to_npz(labels, size):
    pass

# ANCHORHTML = Path("./data/swde/sourceCode/sourceCode")
# ANCHORGRAPHS = Path("./data/swde_HTMLgraphs")
# TARGETFOLDER = Path("movie/movie/movie-allmovie(2000)")
# JSONFILE = "./data/swde_expanded_dataset/dataset/movie/movie-allmovie(2000).json"

# jsonContent = load_json(JSONFILE)

# htmlFolder = ANCHORHTML / TARGETFOLDER
# html_files = list(htmlFolder.rglob("*.htm"))
# htmlAPath = ANCHORGRAPHS / TARGETFOLDER / html_files[0].with_suffix("").name / "A.npz"

# print(label_extraction(html_files[0], jsonContent, htmlAPath))

