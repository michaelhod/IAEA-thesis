from lxml import etree, html
import html5lib
import collections
from typing import Dict
from pathlib import Path

def load_html_as_tree(path: str) -> etree._ElementTree:
    """Parse *HTML* with line‑numbers preserved."""
    with open(path, "rb") as fh:
        raw_bytes = fh.read()

    # html5lib fixes the html
    tree = html5lib.parse(
        raw_bytes,
        treebuilder="lxml",
        namespaceHTMLElements=False,
    )

    _prune_unwanted(tree)
    return tree

def _prune_unwanted(tree: etree._ElementTree) -> None:
    """Remove any element (and its subtree) whose tag is **not** in ALLOWED_TAGS."""
    root = tree.getroot()
    # Walk the tree *bottom‑up* so removing a child doesn't disturb iteration
    for elem in list(reversed(list(root.iter()))):
        if elem.tag not in ALLOWED_TAGS:
            parent = elem.getparent()
            if parent is not None:
                parent.remove(elem)

def iter_elements_with_direct_text(tree: etree._ElementTree):
    """Yield elements that have *direct* (not descendant) text."""
    for elem in tree.iter():
        if elem.text and elem.text.strip():
            yield elem, elem.text
        else:
            for child in elem:
                if child.tail and child.tail.strip():
                    yield elem, child.tail
                    break

def bfs_index_map(tree: etree._ElementTree) -> Dict[etree._Element, int]:
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

def build_parent_and_depth_maps(tree: etree._ElementTree):
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

def compute_hops(a: etree._Element, b: etree._Element,
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

def save_tree_html(tree: etree._ElementTree, out_file: Path | str) -> None:
    """Write the current tree to *out_file* as pretty‑printed HTML.

    Useful during debugging to inspect exactly what remains after pruning.
    """
    out_path = Path(out_file)
    html_str = etree.tostring(
        tree.getroot(),
        pretty_print=True,
        encoding="unicode",
        method="html",
    )
    out_path.write_text(html_str, encoding="utf-8")

# Debugging helpers --------------------------------------------------------------------------------------------------------------------

def debug_dump(tree: etree._ElementTree, max_bytes=10_000):
    """
    Pretty-print the current tree (UTF-8) and truncate after *max_bytes*
    so your debugger console doesn’t blow up.
    """
    html_str = etree.tostring(
        tree,               # ElementTree or root element
        pretty_print=True,  # add line breaks / indent
        encoding="unicode"  # return str, not bytes
    )
    print(html_str[:max_bytes] + ("…[truncated]" if len(html_str) > max_bytes else ""))