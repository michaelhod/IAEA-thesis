# %%
import sys
sys.path.insert(1, r"/vol/bitbucket/mjh24/IAEA-thesis/Stage1/ExtractingGraphs/tagsOfInterest.json")
from Stage1.GAT.GATModel import GraphAttentionNetwork
from Single_Website_Download.Download import main as downloadHTML
import torch
from Stage1.ExtractingGraphs.HTMLtoGraph import html_to_graph, EdgeFeatures
from Stage1.ExtractingGraphs.seleniumDriver import *
from Stage1.ExtractingLabels.swde_label_extraction import label_extraction, _find_matches
from Stage2.txtGraphExtraction.textExtractor import extract_chunk_xpaths
from Stage1.tree_helpers import *
from pathlib import Path
from scipy import sparse
import numpy as np
from collections import defaultdict
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(htmlFilePath, model, safeurl="", specific_node_txt=[], alreadyConvertedToGraph="", ranking_type="min", remove_dupes=True):
    """
    Returns predicted edges given htmlFilePath and model
    URL is used to remove external links from predictions

    Can also search for a specific node given text

    returns a ranked list of predictions from all text nodes as node number, xpath and text
    """

    # %%
    if len(alreadyConvertedToGraph) > 0:
        X_sparse = sparse.load_npz(alreadyConvertedToGraph+"\\X.npz").tocsr()
        X_npy = X_sparse.toarray()
        E_sparse = sparse.load_npz(alreadyConvertedToGraph+"\\E.npz").tocsr()

        df = pd.read_csv(alreadyConvertedToGraph+"\\bbox.csv", index_col=0)
        bbox = df.to_dict(orient="index")
        edge_index = np.load(alreadyConvertedToGraph + "\\edge_index.npy")
    else:
        driver_init(True)
        restart_Driver(True)
        _, X_npy, E_sparse, edge_index, bbox = html_to_graph(htmlFilePath, get_Driver())
        quit_driver()
        X_sparse = sparse.csr_matrix(X_npy)
        E_sparse = sparse.csr_matrix(E_sparse)

    # %%
    def get_all_candidate_edges(filepath, X_npy, safeurl=""):
        xpaths = extract_chunk_xpaths(filepath, safeurl)
        tree = load_html_as_tree(filepath)
        node2index, index2node = bfs_index_map(tree)

        XPaths: Dict[etree._Element, str] = {}
        xp2node: Dict[str, etree._Element] = {}
        for el in node2index:
            xp = tree.getpath(el)
            XPaths[el] = xp
            xp2node[xp] = el
        parentMap, depthMap = build_parent_and_depth_maps(tree)

        txtNodes = []
        coords = []

        for xp in xpaths:
            node = xp2node[xp]
            txtNodes.append(node2index[node])

        #This logic should make an undirected graph
        for i in txtNodes:
            for j in txtNodes:
                if i != j:
                    coords.append((i,j))
        
        features = []
        for i, j in coords:
            features.append(EdgeFeatures(
                edgeStart=i,
                edgeEnd=j,
                edgeStartNode=index2node[i],
                edgeEndNode=index2node[j],
                X=X_npy,
                bboxs=bbox,
                parentMap=parentMap,
                depthMap=depthMap,
                XPaths=XPaths
            ))

        return np.array(coords), features

    # %%
    def get_specific_candidate_edges(filepath, txt, X_npy):
        TXTLENGTH = 400
        tree = load_html_as_tree(filepath)
        node2index, index2node = bfs_index_map(tree)
        parentMap, depthMap = build_parent_and_depth_maps(tree)
        node = _find_matches(tree, txt, depthMap)[0]
        nodeindex = node2index[node]
        
        coords = []

        #This logic should make an undirected graph
        for n, _ in iter_elements_with_direct_text(tree):
            i = node2index[n]
            if i != nodeindex:
                coords.append((i,nodeindex))
                coords.append((nodeindex, i))

        XPaths: Dict[etree._Element, str] = {}
        for el in node2index:
            xp = tree.getpath(el)
            XPaths[el] = xp
        
        features = []
        for i, j in coords:
            features.append(EdgeFeatures(
                edgeStart=i,
                edgeEnd=j,
                edgeStartNode=index2node[i],
                edgeEndNode=index2node[j],
                X=X_npy,
                bboxs=bbox,
                parentMap=parentMap,
                depthMap=depthMap,
                XPaths=XPaths
            ))

        return np.array(coords), features

    # %%
    if len(specific_node_txt) > 0:
        label_index, label_features = get_specific_candidate_edges(htmlFilePath, specific_node_txt, X_npy)
    else:    
        label_index, label_features = get_all_candidate_edges(htmlFilePath, X_npy, safeurl)
    label_features = sparse.csr_matrix(label_features)

    # %%
    def _npz_to_csr(csr: sparse.csr, dtype=torch.float32):
        crow = torch.from_numpy(csr.indptr.astype(np.int64))
        col  = torch.from_numpy(csr.indices.astype(np.int64))
        val  = torch.from_numpy(csr.data).to(dtype)
        return torch.sparse_csr_tensor(
            crow, col, val, size=csr.shape, dtype=dtype, requires_grad=False
        )

    def _npy_to_tensor(arr: np.ndarray, dtype=torch.long):
        return torch.from_numpy(arr).to(dtype).t().contiguous()

    X, Aei, Aef, Lei, Lef = _npz_to_csr(X_sparse), _npy_to_tensor(edge_index), _npz_to_csr(E_sparse), _npy_to_tensor(label_index), _npz_to_csr(label_features)
    X, Aei, Aef, Lei, Lef = X.to(device), Aei.to(device), Aef.to(device), Lei.to(device), Lef.to(device)

    # %%
    model.eval()
    logits = model(X, Aei, Aef, Lei, Lef)
    probs  = torch.sigmoid(logits)

    # %%
    order = np.argsort(probs.squeeze().tolist())[::-1]
    probs = probs.squeeze().detach().numpy()

    sorted_label_index = label_index[order]
    sorted_probs = probs[order]

    # Build a mapping from tuple to its position for fast lookup
    pair_to_pos = {tuple(pair): idx for idx, pair in enumerate(sorted_label_index)}
    avg_pos = []
    for idx, label in enumerate(sorted_label_index):
        rev_pair = (label[1], label[0])
        rev_idx = pair_to_pos[rev_pair]
        if ranking_type == "min":
            ranking = min(idx, rev_idx)
            ranking = ranking if ranking == idx else ranking + 1e-9 #This is so that the lesser of the two is ranked behind
            avg_pos.append(ranking)
        else:
            ranking = (idx + rev_idx) / 2
            ranking = ranking if min(idx, rev_idx) == idx else ranking + 1e-9 #This is so that the lesser of the two is ranked behind
            avg_pos.append(ranking)

    order = np.argsort(avg_pos)
    sorted_label_index = sorted_label_index[order]
    sorted_probs = sorted_probs[order]

    # %%
    tree = load_html_as_tree(htmlFilePath)
    _, index2node = bfs_index_map(tree)

    txts = []
    xpaths = []
    for label in sorted_label_index:
        txts.append([get_node_text(index2node[label[0]]).strip(), get_node_text(index2node[label[1]]).strip()])
        xpaths.append([tree.getpath(index2node[label[0]]), tree.getpath(index2node[label[1]])])

    # remove duplicates. This is needed as the same text chunk can be picked up under different ancestor nodes. Due to the nature of sorted_label_index ordering, the highest probability will be saved
    deduped_label = []
    deduped_xpath = []
    deduped_txts = []
    deduped_probs = []

    def _is_ancestor_or_descendant(a: str, b: str) -> bool:
        return a.startswith(b + '/') or b.startswith(a + '/') or a==b

    # seen maps an ordered text pair -> list of xpath pairs we've already accepted
    seen = {}

    for idx, (coord, xpath_pair, txt_pair, prob) in enumerate(zip(sorted_label_index, xpaths, txts, sorted_probs)):
        u_txt, v_txt = txt_pair
        u_xpath, v_xpath = xpath_pair
        pair = (u_txt, v_txt)

        if normalise_text(u_txt) == normalise_text(v_txt) and _is_ancestor_or_descendant(u_xpath, v_xpath):
            continue

        # Have we seen this ordered text pair with compatible (ancestor/descendant) xpaths?
        def _paths_match_any(seen_pairs):
            for su, sv in seen_pairs:
                if _is_ancestor_or_descendant(u_xpath, su) and _is_ancestor_or_descendant(v_xpath, sv):
                    return True
            return False

        already_seen = False
        if pair in seen and _paths_match_any(seen[pair]):
            already_seen = True
        elif remove_dupes:
            # optionally consider reversed pair as a duplicate too
            rev = (v_txt, u_txt)
            if rev in seen:
                # compare crosswise (current u vs seen v, current v vs seen u)
                for sv, su in seen[rev]:
                    if _is_ancestor_or_descendant(u_xpath, su) and _is_ancestor_or_descendant(v_xpath, sv):
                        already_seen = True
                        break

        # add to deduped
        if not already_seen:
            deduped_label.append(coord)
            deduped_xpath.append([u_xpath, v_xpath])
            deduped_txts.append([u_txt, v_txt])
            deduped_probs.append(prob)
            seen.setdefault(pair, []).append((u_xpath, v_xpath))

    return deduped_label, deduped_xpath, deduped_txts, deduped_probs

def keepTopKMask(arr, k: int):
    """arr is shape (K,2), ordered. It keeps the first k instances of each individual instance"""

    uniqueValuesInArr = np.unique(arr, axis=None)
    arr = np.array(arr)
    mask = np.array([False]*len(arr))

    for uniquevalue in uniqueValuesInArr:
        i = 0
        for idx, pair in enumerate(arr):
            if i >= k:
                break
            if uniquevalue in pair:
                i+=1
                mask[idx] = True

    return mask

def filterTextMask(textArr, filter, exact=True):
    mask = np.array([False]*len(textArr))
    if exact:
        for idx, pair in enumerate(textArr):
            if filter in pair:
                mask[idx] = True
    else:
        for idx, pair in enumerate(textArr):
            if filter in pair[0] or filter in pair[1]:
                mask[idx] = True
    return mask

if __name__ == "__main__":
    #Import model
    # model = GraphAttentionNetwork(in_dim = 119, edge_in_dim = 210, edge_emb_dim = 32, hidden1 = 32, hidden2 = 32, hidden3 = 8, heads = 2)
    # state_dict = torch.load("./Stage1/GAT/FULLTRAINEDALLDATAModelf1-83-newtagsnotitle.pt", map_location=torch.device(device))
    # model = GraphAttentionNetwork(in_dim = 114, edge_in_dim = 200, edge_emb_dim = 32, hidden1 = 32, hidden2 = 32, hidden3 = 8, heads = 2)
    # state_dict = torch.load("./Stage1/GAT/FULLTRAINEDALLDATAModelf1-75-learning.pt", map_location=torch.device(device))
    model = GraphAttentionNetwork(in_dim = 119, edge_in_dim = 210, edge_emb_dim = 32, hidden1 = 32, hidden2 = 32, hidden3 = 8, heads = 2)
    state_dict = torch.load("./Stage1/GAT/LONG80EPOCH-75f1-newlabelnotitle.pt", map_location=torch.device(device))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # url = "C:\\Users\\micha\\Documents\\Imperial Courses\\Thesis\\IAEA-thesis\\data\\swde\\sourceCode\\sourceCode\\movie\\movie\\movie-allmovie(2000)\\0000.htm"
    # url = r"https://www.nucnet.org/news/parliament-resolution-paves-way-for-establishing-nuclear-energy-legislation-6-4-2024"
    # url = "https://westinghousenuclear.com/"
    # url = "https://www.football.co.uk/news/leeds-vs-bournemouth-premier-league-team-news-lineups-prediction/781112/"
    # url = r"https://www.bbc.co.uk/news/live/cev28rvzlv1t"
    url = "https://www.nfl.com/teams/" # Great to show teams and structured data
    # url = "https://www.energy.gov/ne/articles/advantages-and-challenges-nuclear-energy" #Great to show semi structured webpages with titles
    # url = "https://westinghousenuclear.com/nuclear-fuel/fuel-fabrication-operations/"
    # url = "https://www.livescore.com/en/football/england/premier-league/bournemouth-vs-leicester-city/1250940/lineups/"
    htmlFile = Path("C:/Users/micha/Documents/Imperial Courses/Thesis/IAEA-thesis/data/websites/test.html")
    downloadHTML(url,1,htmlFile)

    sorted_label_index, xpaths, txts, probs = main(htmlFile, model, remove_dupes=False)
    normtxt = []
    for a, b in txts:
        normtxt.append([normalise_text(a), normalise_text(b)])
    txts = np.array(normtxt)
    xpaths = np.array(xpaths)
    sorted_label_index = np.array(sorted_label_index)
    probs = np.array(probs)
    #mask = keepTopKMask(txts, 1)
    mask = filterTextMask(txts, "afcteams", False)#13karrizabalaga
    
    for row in zip(sorted_label_index[mask][:200], xpaths[mask][:200], txts[mask][:200], probs[mask][:200]):
        print(row[2])
        print("\t", row[3])
        print("\t", row[0])
        print("\t", row[1])
    
    #xpaths = np.array(xpaths)
    #print(txts[mask][:200])
    #print(probs[mask][:200])
