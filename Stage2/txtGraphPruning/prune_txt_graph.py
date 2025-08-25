import sys
sys.path.insert(1, r"C:\\Users\\micha\\Documents\\Imperial Courses\\Thesis\\IAEA-thesis")
from Stage1.GAT.GATModel import GraphAttentionNetwork
from Single_Website_Download.Download import main as downloadHTML
import torch
from Stage1.ExtractingGraphs.seleniumDriver import *
from Stage2.txtGraphExtraction.extract_mini_txt_graphs_helper import *
from Stage2.txtGraphExtraction.extract_mini_txt_graphs_helper import main as txtExtractor
from Stage1.tree_helpers import *
from Stage2.txtGraphPruning.display_txt_graph import draw_graph_from_arrays
from pathlib import Path
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def disparity_backbone_bh(
    P: np.ndarray,
    q: float = 0.05,
    side: str = "out",          # "out" | "in" | "or" | "and"
    keep_self_loops: bool = False,
    fallback_out: int = 1,             # keep top-k outgoing if none survived
    fallback_in: int = 1,              # keep top-k incoming if none survived
):
    W = np.asarray(P, dtype=float).copy()
    if np.any(W < 0):
        raise ValueError("P must be non-negative.")
    N = W.shape[0]
    if not keep_self_loops:
        np.fill_diagonal(W, 0.0)

    M = W > 0  # edges

    # ---------- OUTGOING side ----------
    s_out = W.sum(axis=1, keepdims=True)                     # (N,1)
    p_out = np.divide(W, s_out, out=np.zeros_like(W), where=s_out > 0)
    k_out = M.sum(axis=1, keepdims=True)                     # (N,1)
    exp_out = np.maximum(k_out - 1, 1)                       # (N,1), avoid 0
    # compute α_out everywhere, default 1 for non-edges
    alpha_out = np.ones_like(W)
    np.power(1.0 - p_out, exp_out, where=M, out=alpha_out)
    # rows with exactly one outgoing edge: force α_out=0 on that edge
    for r in np.where(k_out.ravel() == 1)[0]:
        alpha_out[r, M[r]] = 0.0

    # ---------- INCOMING side ----------
    s_in = W.sum(axis=0, keepdims=True)                      # (1,N)
    p_in = np.divide(W, s_in, out=np.zeros_like(W), where=s_in > 0)
    k_in = M.sum(axis=0, keepdims=True)                      # (1,N)
    exp_in = np.maximum(k_in - 1, 1)                         # (1,N)
    alpha_in = np.ones_like(W)
    np.power(1.0 - p_in, exp_in, where=M, out=alpha_in)
    # cols with exactly one incoming edge: force α_in=0 on that edge
    for c in np.where(k_in.ravel() == 1)[0]:
        alpha_in[M[:, c], c] = 0.0

    # ---------- BH helper ----------
    def bh_mask_1d(pvals: np.ndarray, q: float) -> np.ndarray:
        m = pvals.size
        if m == 0:
            return np.zeros(0, dtype=bool)
        order = np.argsort(pvals)
        pv_sorted = pvals[order]
        thresh = (np.arange(1, m + 1) / m) * q
        passed = pv_sorted <= thresh
        if not np.any(passed):
            return np.zeros(m, dtype=bool)
        k = np.where(passed)[0].max()
        cutoff = pv_sorted[k]
        return pvals <= cutoff

    # ---------- apply BH per row/col ----------
    keep_out = np.zeros_like(M)
    keep_in  = np.zeros_like(M)

    if side in ("out", "or", "and"):
        for i in range(N):
            idx = np.where(M[i])[0]
            if idx.size:
                mask = bh_mask_1d(alpha_out[i, idx], q)
                keep_out[i, idx[mask]] = True

    if side in ("in", "or", "and"):
        for j in range(N):
            idx = np.where(M[:, j])[0]
            if idx.size:
                mask = bh_mask_1d(alpha_in[idx, j], q)
                keep_in[idx[mask], j] = True

    if side == "out":
        keep = keep_out
    elif side == "in":
        keep = keep_in
    elif side == "or":
        keep = keep_out | keep_in
    elif side == "and":
        keep = keep_out & keep_in
    else:
        raise ValueError("side must be one of {'out','in','or','and'}")

    if not keep_self_loops:
        np.fill_diagonal(keep, False)

    # ---------- Fallbacks: guarantee some edges survive ----------
    # Outgoing fallback: if a row has no kept edges, keep its top-k outgoing existing edges.
    if fallback_out > 0:
        for i in range(N):
            if keep[i].any():
                continue
            # candidate outgoing (existing, no self-loop if disabled)
            cand = M[i].copy()
            if not keep_self_loops:
                cand[i] = False
            if not cand.any(): # Is there anything at all to keep, or is the original node isolated too
                continue
            k = min(fallback_out, cand.sum())
            # indices of top-k by weight in row i
            j_idx = np.argpartition(W[i, cand], -k)[-k:]
            cols = np.flatnonzero(cand)[j_idx]
            keep[i, cols] = True

    # Incoming fallback: if a column has no kept edges, keep its top-k incoming existing edges.
    if fallback_in and fallback_in > 0:
        for j in range(N):
            if keep[:, j].any():
                continue
            cand = M[:, j].copy()
            if not keep_self_loops:
                cand[j] = False
            if not cand.any():
                continue
            k = min(fallback_in, cand.sum())
            i_idx = np.argpartition(W[cand, j], -k)[-k:]
            rows = np.flatnonzero(cand)[i_idx]
            keep[rows, j] = True

    ii, jj = np.nonzero(keep)
    edges = np.column_stack((ii, jj))
    weights = W[keep]
    return edges, weights, keep

def clip_topk(W: np.ndarray, k: int, axis: int = 1) -> np.ndarray:
    """
    Keep the top-k *nonzero* entries along the given axis; zero out the rest.
    axis=1 → per row (outgoing), axis=0 → per column (incoming).
    Stable for ties: keeps arbitrary k among equals (argpartition's behavior).
    """
    W = np.asarray(W, dtype=float)
    if k <= 0:
        return np.zeros_like(W)

    if axis == 1:
        # work row-wise
        out = np.zeros_like(W)
        # mask of nonzero entries per row
        nz_counts = (W != 0).sum(axis=1)
        for i in range(W.shape[0]):
            mi = nz_counts[i]
            if mi == 0:
                continue
            kk = min(k, mi)
            # indices of top-k in row i
            idx = np.argpartition(W[i], -kk)[-kk:]
            out[i, idx] = W[i, idx]
        return out

    elif axis == 0:
        # work column-wise
        out = np.zeros_like(W)
        nz_counts = (W != 0).sum(axis=0)
        for j in range(W.shape[1]):
            mj = nz_counts[j]
            if mj == 0:
                continue
            kk = min(k, mj)
            idx = np.argpartition(W[:, j], -kk)[-kk:]
            out[idx, j] = W[idx, j]
        return out
    else:
        raise Exception("Axis is either 1 or 0")
    
def main(probs, sorted_label_index):
    
    # --- Build probability matrix ---
    max_id = int(sorted_label_index.max())
    n = max_id + 1  # beware if IDs are very sparse -> big matrix

    P = np.zeros((n, n), dtype=float)
    for (u, v), p in zip(sorted_label_index, probs):
        P[int(u), int(v)] = float(p)
        if REMOVE_DUPES:
            P[int(v), int(u)] = float(p)

    P_clipped = clip_topk(P, CLIP_TO_N_EDGES_PER_NODE)

    # --- Filter out edges ---
    edges, weights, keep = disparity_backbone_bh(P_clipped, 0.05, "out", False, 2, 2)
    mask = [keep[i][j] for i,j in sorted_label_index]
    if len(mask) != len(sorted_label_index):
        raise Exception("The mask is a different length. Something is wrong")
    
    return mask

if __name__ == "__main__":
    CLIP_TO_N_EDGES_PER_NODE = 10

    # Load model
    model = GraphAttentionNetwork(in_dim = 119, edge_in_dim = 210, edge_emb_dim = 32, hidden1 = 32, hidden2 = 32, hidden3 = 8, heads = 2)
    state_dict = torch.load("./Stage1/GAT/LONG80EPOCH-75f1-newlabelnotitle.pt", map_location=torch.device(device))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)

    # Load the website
    url = "https://www.nfl.com/teams/"
    htmlFile = Path("C:/Users/micha/Documents/Imperial Courses/Thesis/IAEA-thesis/data/websites/test.html")
    downloadHTML(url,1,htmlFile)

    # Create label edges from text nodes in the website
    REMOVE_DUPES=False
    sorted_label_index, xpaths, txts, probs = txtExtractor(htmlFile, model, remove_dupes=REMOVE_DUPES)

    normtxt = []
    for a, b in txts:
        normtxt.append([normalise_text(a, "'\\s"), normalise_text(b, "'\\s")])
    txts, probs, sorted_label_index, xpaths = np.array(normtxt), np.array(probs), np.array(sorted_label_index), np.array(xpaths)

    # -- RUN THE MAIN PRUNING MASK --
    mask = main(probs, sorted_label_index)

    # Concatanate and apply masks if we want specific text
    mask = mask #& filterTextMask(txts, "pittsburghsteelers", False) #& mask = keepTopKMask(txts, 1)
    print(len(sorted_label_index), " -> ", len(sorted_label_index[mask]))
    txts, probs, sorted_label_index, xpaths = txts[mask], probs[mask], sorted_label_index[mask], xpaths[mask]
    # -- PRUNING FINISHED --

    for row in zip(sorted_label_index[:200], xpaths[:200], txts[:200], probs[:200]):
        print(row[2])
        # print("\t", row[3])
        # print("\t", row[0])
        # print("\t", row[1])

    draw_graph_from_arrays(
        txt_edge_pairs=txts,
        id_edge_pairs=sorted_label_index,
        xpath_edge_pairs=xpaths,
        probs=probs,
        title="Graph",
    )