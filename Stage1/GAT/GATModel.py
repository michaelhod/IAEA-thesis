# To advance the model, use the methods in https://arxiv.org/pdf/2311.02921
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
import torch.nn.functional as F

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
                 edge_in_dim: int   = 197,
                 edge_emb_dim: int  = 8,
                 hidden1: int       = 128,
                 hidden2: int       = 64,
                 hidden3: int       = 32,
                 heads:  int        = 4):
        super().__init__()

        # ── Node-level encoder (edge-aware) ────────────────────────────
        self.tr1 = TransformerConv(
            in_channels      = in_dim,
            out_channels     = hidden1,
            heads            = heads,
            edge_dim         = edge_emb_dim,
            dropout          = 0.1,
            beta             = False         # learnable α in α·x + (1-α)·attn
        )
        self.ln1 = nn.LayerNorm(hidden1 * heads)

        self.tr2 = TransformerConv(
            in_channels      = hidden1 * heads,
            out_channels     = hidden2,
            heads            = 1,
            edge_dim         = edge_emb_dim,
            dropout          = 0.1,
            beta             = False
        )
        self.ln2 = nn.LayerNorm(hidden2)
        self.tr3 = TransformerConv(
            in_channels      = hidden2,
            out_channels     = hidden3,
            heads            = 1,
            edge_dim         = edge_emb_dim,
            dropout          = 0.1,
            beta             = False
        )

        # ── Edge feature projector ────────────── (It is not an explicit linear layer as it works on a sparse matrix)
        self.AW_edge = nn.Parameter(torch.empty(edge_in_dim, edge_emb_dim))
        nn.init.xavier_uniform_(self.AW_edge)
        # self.EW_edge = nn.Parameter(torch.empty(edge_in_dim, edge_emb_dim))
        # nn.init.xavier_uniform_(self.EW_edge)

        # ── Edge-level MLP decoder (unchanged) ────────────────────────
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden3 * 2 + edge_emb_dim, hidden3),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.Linear(hidden3, 1)
        )

        # ── Node (title) head ─────────────────────────────────────────────
        # Scores each node independently; softmax over nodes is applied in the loss.
        self.title_head = nn.Sequential(
            nn.Linear(hidden3, hidden3),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden3, 1)
        )

        # init beta gate around 0.5 to avoid identity lock
        for tr in (self.tr1, self.tr2):
            if getattr(tr, "lin_beta", None) is not None:
                nn.init.zeros_(tr.lin_beta.weight)
                if tr.lin_beta.bias is not None:
                    nn.init.zeros_(tr.lin_beta.bias)

    # ---------------------------------------------------------------------

    def forward(
        self,
        x_sparse: torch.Tensor,        # (N_nodes, 96)          sparse
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
        h = F.relu( self.tr1(x_dense, A_edge_index, A_edge_emb) )
        #h = self.ln1(h)
        #Try a linlayer here to condense heads
        h = F.relu( self.tr2(h,        A_edge_index, A_edge_emb) )
        #h = self.ln2(h)
        h = F.relu( self.tr3(h,        A_edge_index, A_edge_emb) )

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