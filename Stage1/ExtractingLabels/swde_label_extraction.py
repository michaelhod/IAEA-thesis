import json
from pathlib import Path
from typing import Dict
from lxml import etree, html
import collections

TAGSOFINTEREST = json.load(open("Stage1/ExtractingGraphs/tagsOfInterest.json", "r"))
ALLOWED_TAGS = set(TAGSOFINTEREST.keys())

def _prune_unwanted(tree: etree._ElementTree) -> None:
    """Remove any element (and its subtree) whose tag is **not** in ALLOWED_TAGS."""
    root = tree.getroot()
    # Walk the tree *bottom‑up* so removing a child doesn't disturb iteration
    for elem in list(reversed(list(root.iter()))):
        if elem.tag not in ALLOWED_TAGS:
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)

def load_html_as_tree(path: str) -> etree._ElementTree:
    """Parse *HTML* with line‑numbers preserved."""
    parser = etree.HTMLParser(huge_tree=True)
    tree = html.parse(path, parser)
    _prune_unwanted(tree)
    return tree

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

def _iter_elements_with_direct_text(tree: etree._ElementTree):
    """Yield elements that have *direct* (not descendant) text."""
    for elem in tree.iter():
        if elem.text and elem.text.strip():
            yield elem, elem.text
        else:
            for child in elem:
                if child.tail and child.tail.strip():
                    yield elem, child.tail
                    break

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
    direct_matches = [el for el, part_text in _iter_elements_with_direct_text(tree)
                      if needle_lower in part_text.lower()]  # type: ignore[arg-type]
    if direct_matches:
        return direct_matches

    # 2. Fallback: elements whose text is a *part* of the needle
    substr_candidates = [(el, part_text) for el, part_text in _iter_elements_with_direct_text(tree)
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

def _bfs_index_map(tree: etree._ElementTree) -> Dict[etree._Element, int]:  # type: ignore[name-defined]
    """Return a mapping ``element → BFS index`` (level‑order numbering)."""
    root = tree.getroot()
    queue = collections.deque([root])
    index_map: Dict[etree._Element, int] = {}
    idx = 0
    while queue:
        node = queue.popleft()
        index_map[node] = idx
        idx += 1
        queue.extend(list(node))
    return index_map

def _build_parent_and_depth_maps(tree: etree._ElementTree):
    """
    Build maps of node→parent and node→depth for all elements in the tree.
    Root has parent None and depth 0.
    """
    root = tree.getroot()
    parent_map: Dict[etree._Element, etree._Element] = {}
    depth_map: Dict[etree._Element, int] = {root: 0}
    # BFS to establish parent and depth
    queue: collections.deque[etree._Element] = collections.deque([root])  # type: ignore[name-defined]
    while queue:
        node = queue.popleft()
        for child in list(node):
            parent_map[child] = node
            depth_map[child] = depth_map[node] + 1
            queue.append(child)
    return parent_map, depth_map


def _compute_hops(a: etree._Element, b: etree._Element,
                  parent_map: Dict[etree._Element, etree._Element],
                  depth_map: Dict[etree._Element, int]) -> int:
    """
    Compute number of hops (edges) between nodes a and b in the tree.
    """
    # Bring a and b to the same depth
    da, db = depth_map.get(a, 0), depth_map.get(b, 0)
    hops = 0
    na, nb = a, b
    # Ascend deeper node
    while da > db:
        na = parent_map.get(na, na)
        da -= 1; hops += 1
    while db > da:
        nb = parent_map.get(nb, nb)
        db -= 1; hops += 1
    # Ascend both until common ancestor
    while na is not nb:
        na = parent_map.get(na, na)
        nb = parent_map.get(nb, nb)
        hops += 2
    return hops

def _closest_for_pair(tree: etree._ElementTree, left: str, right: str):
    """Compute closest (by line diff) element pair for *left*, *right*."""
    parent_map, depth_map = _build_parent_and_depth_maps(tree)
    
    nodes_left = _find_matches(tree, left, depth_map)
    nodes_right = _find_matches(tree, right, depth_map)

    if not nodes_left or not nodes_right:
        # No match for one or both texts
        return f"{left} | {right}"

    bfs_indices = _bfs_index_map(tree)

    best_hops = float("inf")
    best_pair = (None, None)
    for a in nodes_left:
        for b in nodes_right:
            hops = _compute_hops(a, b, parent_map, depth_map)
            if hops < best_hops:
                best_hops = hops
                best_pair = (a, b)

    return (bfs_indices[best_pair[0]], bfs_indices[best_pair[1]])

def label_extraction(htmlFile: Path, jsonContent) -> None:

    tree = load_html_as_tree(htmlFile)
    htmlName = htmlFile.name

    results = [_closest_for_pair(tree, left, right) for left, right in iterate_pairs(jsonContent, htmlName)]

    return results

def labels_to_npz(labels, size):
    pass


jsonContent = load_json("./data/swde_expanded_dataset/dataset/movie/movie-allmovie(2000).json")

htmlFolder = Path("./data/swde/sourceCode/sourceCode/movie/movie/movie-allmovie(2000)")
html_files = list(htmlFolder.rglob("*.htm"))

labels_to_npz(label_extraction(html_files[0], jsonContent))

