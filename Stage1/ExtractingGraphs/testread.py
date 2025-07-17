from scipy import sparse
import torch
import numpy as np

A_csr = sparse.load_npz("./data/swde_HTMLgraphs/movie/test/A.npz")  # Just change A.npz for E or X
# Convert to COO:
A_coo = A_csr.tocoo()

# Build a PyTorch sparse tensor
indices = torch.vstack([
    torch.from_numpy(A_coo.row.astype(np.int64)),
    torch.from_numpy(A_coo.col.astype(np.int64))
])
values = torch.from_numpy(A_coo.data)
print(A_coo.shape)
print(A_coo)

print(np.load("./data/swde_HTMLgraphs/movie/test/edge_index.npy"))