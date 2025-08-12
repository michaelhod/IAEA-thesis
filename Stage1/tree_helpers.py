import html5lib._ihatexml
from lxml import etree
import html
import collections
from typing import Dict
from pathlib import Path
import json
import re
import warnings, html5lib

warnings.filterwarnings(
    "ignore",
    category=html5lib.serializer.DataLossWarning
    if hasattr(html5lib.serializer, "DataLossWarning")
    else html5lib._ihatexml.DataLossWarning
)

TAGSOFINTEREST = json.load(open("C:/Users/micha/Documents/Imperial Courses/Thesis/IAEA-thesis/Stage1/ExtractingGraphs/tagsOfInterest.json", "r"))
ALLOWED_TAGS = set(TAGSOFINTEREST.keys())

_ILLEGAL_CTRL = re.compile(r'&#(?:0*([0-9]+)|x0*([0-9a-fA-F]+));')

def _strip_bad_refs(text: str) -> str:
    """Remove numeric character refs that map to disallowed C0 controls."""
    def repl(m):
        dec = m.group(1)
        value = int(dec, 10) if dec else int(m.group(2), 16)
        return '' if 0 <= value < 32 and value not in (9, 10, 13) else m.group(0)
    return _ILLEGAL_CTRL.sub(repl, text)

def load_htmlstr_as_tree(html: str) -> etree._ElementTree:
    """Parse html string"""
    html = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', html)
    html = _strip_bad_refs(html)

    tree = html5lib.parse(
        html,
        treebuilder="lxml",
        namespaceHTMLElements=False,
    )

    _prune_unwanted(tree)
    return tree

def load_html_as_tree(path: Path) -> etree._ElementTree:
    """Parse path"""
    with open(path, "rb") as fh:
        raw_bytes = fh.read()

    text = raw_bytes.decode("utf-8", "replace")

    # 2) drop literal C0 bytes AND refs like &#27;
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)
    text = _strip_bad_refs(text)

    # Second pass: build the *lxml* tree
    tree = html5lib.parse(
        text,
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
        for child in elem: # Check the rest of the element
            if child.tail and child.tail.strip():
                yield elem, child.tail
                # break

def bfs_index_map(tree: etree._ElementTree) -> tuple[Dict[etree._Element, int], list[etree._Element]]:
    """Return a mapping ``element → BFS index`` (level‑order numbering)."""
    root = tree.getroot()
    queue = collections.deque([root])
    index_map: Dict[etree._Element, int] = {}
    node_map = []
    idx = 0
    while queue:
        node = queue.popleft()
        index_map[node] = idx
        node_map.append(node)
        idx += 1
        queue.extend(list(node))
    return index_map, node_map

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

def get_node_text(node: etree._Element, normalised=False) -> str:
    txt = ''.join(node.itertext()).strip()
    if normalised:
        txt = normalise_text(txt)
    return txt

def normalise_text(txt:str) -> str:
    txt = html.unescape(txt).lower().replace("\r","\n") # normaluse carridge return
    txt = re.sub(r"[^a-z0-9]+", "", txt.lower())
    return txt

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