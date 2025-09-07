# To advance the model, use the methods in https://arxiv.org/pdf/2311.02921
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GPSConv, GINEConv

class GraphAttentionNetwork(nn.Module):
    """
    HTML‑graph model

        X  ─╮
            │  GAT( 96 → 64 )
            │  ReLU
            │  GAT( 64 → 32 )
            │  ReLU
            └─ Edge‑feature constructor
                      [h_i ‖ h_j ‖ φ(e_ij)] ─► MLP(69 → 1)

    Parameters
    ----------
    in_dim          : node‑feature size   (= 96)
    edge_in_dim     : raw edge‑feature size (= 197)
    edge_emb_dim    : Edge-feature MLP output dims
    """
    def __init__(self,
                 in_dim: int        = 96,
                 pe_dim: int        = 10,
                 edge_in_dim: int   = 197,
                 edge_emb_dim: int  = 8,
                 heads:  int        = 4):
        super().__init__()

        self.pe_lin = nn.Linear(18, pe_dim)
        self.pe_norm = nn.BatchNorm1d(18)

        self.convs = nn.ModuleList()
        channels = in_dim - (18-pe_dim)
        for _ in range(6):
            seq = nn.Sequential(
                nn.Linear(channels, channels),
                nn.ReLU(),
                nn.Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(seq, edge_dim=edge_emb_dim), heads=heads)
            self.convs.append(conv)

        # ── Edge feature projector ────────────── (It is not an explicit linear layer as it works on a sparse matrix)
        self.AW_edge = nn.Parameter(torch.empty(edge_in_dim, edge_emb_dim))
        nn.init.xavier_uniform_(self.AW_edge)
        # self.EW_edge = nn.Parameter(torch.empty(edge_in_dim, edge_emb_dim))
        # nn.init.xavier_uniform_(self.EW_edge)

        # ── Edge-level MLP decoder (unchanged) ────────────────────────
        self.edge_mlp = nn.Sequential(
            nn.Linear(channels * 2 + edge_emb_dim, channels),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(channels, 1)
        )

        # ── Node (title) head ─────────────────────────────────────────────
        # Scores each node independently; softmax over nodes is applied in the loss.
        self.title_head = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(channels, 1)
        )

    # ---------------------------------------------------------------------

    def forward(
        self,
        x_sparse: torch.Tensor,        # (N_nodes, 96)          sparse
        batch,
        A_edge_index: torch.Tensor,   # (2, nnz_A)             COO  (from A)
        A_edge_attr: torch.Tensor,    # (nnz_A, 197)           dense / sparse.mm
        E_edge_index: torch.Tensor,   # (2, N_E)               candidates
        E_edge_attr: torch.Tensor,    # (N_E, 197)             sparse features
        E_attr_dropout=0.0,            # Probability of dropping out a whole edge_attr when training
        E_attr_include=True,
        A_attr_include=True,
        return_title = False
    ):
        # 1) node features
        x_dense = x_sparse.to_dense()
        A_edge_emb = torch.sparse.mm(A_edge_attr, self.AW_edge)     # (nnz_A , 8)
        #A_edge_emb = A_edge_attr.to_dense()
        #E_edge_emb = E_edge_attr.to_dense()

        if not A_attr_include:
            A_edge_emb = torch.zeros_like(A_edge_emb)

        # 2) edge-aware GATv2 layers
        pe = x_dense[:,-18:]
        x = x_dense[:,:-18]
        x_pe = self.pe_norm(pe)
        h = torch.cat((x, self.pe_lin(x_pe)), 1)

        for conv in self.convs:
            h = conv(h, A_edge_index, batch, edge_attr=A_edge_emb)

        # 3) candidate-edge projection  φ(E) = E @ W_edge
        E_edge_emb = torch.sparse.mm(E_edge_attr, self.AW_edge)     # (N_E , 8)
        
        if self.training:
            mask = torch.rand(E_edge_emb.size(0), 1,
                            device=E_edge_emb.device) > E_attr_dropout   # (N_E,1)
            E_edge_emb = E_edge_emb * mask

        if not E_attr_include:
            E_edge_emb = torch.zeros_like(E_edge_emb)

        # 4) gather node embeddings and classify
        src, dst = E_edge_index
        z = torch.cat([h[src], h[dst], E_edge_emb], dim=1)      # (N_E , 72)
        edge_logits = self.edge_mlp(z).squeeze(-1)                   # (N_E ,) returns the logits
    
        # 5) Title node scores (one scalar per node)
        title_logits = self.title_head(h).squeeze(-1)  # (N,)

        # Backward-compatible: only return title logits if asked
        # Add a keyword arg to the signature: return_title: bool = False
        return (edge_logits, title_logits) if return_title else edge_logits