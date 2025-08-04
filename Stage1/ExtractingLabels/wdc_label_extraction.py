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

            if parent is not None:
                parent_idx = nodeToIndex[parent]
                toLink[parent_idx].append(idx)

    for parent in toLink:
        toLink[parent].append(parent)

    # now build all unordered pairs among the siblings of each parent
    pairs = []
    for child_indices in toLink.values():
        # need at least two children to make a pair
        if len(child_indices) >= 2:
            pairs.extend(itertools.combinations(child_indices, 2))

    return pairs


def label_extraction(htmlFile: Path, dataPath:Path, save=False, verifyTreeAgainstFile=False, displayLabels=False, displaynegativeLabels=False) -> list[tuple]:

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
        _save_coords_to_npz(label_index, label_features, label_value, dataPath)

    return results, label_index, label_features, label_value