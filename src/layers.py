import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GraphTransformerLayer(nn.Module):
    """
    High-Speed Multi-Head Graph Attention Layer with Edge Features,
    Residual Connections, Layer Normalization, and Feed-Forward Network.
    Optimized for rapid CPU tensor operations.
    """
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, num_edge_types: int = 11, dropout: float = 0.15):
        super(GraphTransformerLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        
        # Edge type embeddings
        self.edge_emb = nn.Embedding(num_edge_types + 2, in_dim)
        
        # High-Speed Multi-Head Attention Convolution
        self.conv = GATv2Conv(
            in_channels=in_dim,
            out_channels=out_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=in_dim,
            dropout=dropout,
            share_weights=True
        )
        
        self.norm1 = nn.LayerNorm(out_dim)
        self.norm2 = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim),
            nn.Dropout(dropout)
        )
        
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        x: [N, in_dim]
        edge_index: [2, E]
        edge_attr: [E]
        """
        res = self.res_proj(x)
        
        if edge_index is not None and edge_index.numel() > 0 and edge_index.size(1) > 0:
            if edge_attr is not None and edge_attr.numel() > 0:
                e_emb = self.edge_emb(edge_attr.clamp(0, self.num_edge_types))
            else:
                e_emb = torch.zeros((edge_index.size(1), self.in_dim), device=x.device)
            h = self.conv(x, edge_index, edge_attr=e_emb)
        else:
            h = res
            
        x = self.norm1(res + self.dropout(h))
        x = self.norm2(x + self.ffn(x))
        return x
