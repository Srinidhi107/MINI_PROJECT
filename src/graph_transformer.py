import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

from src.layers import GraphTransformerLayer
from src.pooling import get_pooling_layer

def drop_edge(edge_index: torch.Tensor, edge_attr: torch.Tensor, drop_rate: float = 0.15, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Randomly drop edges during training for graph regularization."""
    if not training or drop_rate <= 0.0 or edge_index.size(1) == 0:
        return edge_index, edge_attr
    mask = torch.rand(edge_index.size(1), device=edge_index.device) > drop_rate
    # Ensure at least some edges remain
    if mask.sum() == 0:
        return edge_index, edge_attr
    return edge_index[:, mask], edge_attr[mask] if edge_attr.numel() > 0 else edge_attr

class NodeFeatureEncoder(nn.Module):
    """
    Rich Multi-Modal Node Feature Encoder:
    1. Node type projection (17 categories)
    2. Code token embeddings with padding mask (10 tokens)
    3. Structural Degree Centrality (In-degree & Out-degree embeddings)
    4. Node-level security & syntax risk indicators (6 dims)
    """
    def __init__(
        self,
        vocab_size: int,
        node_type_dim: int = 17,
        max_tokens: int = 10,
        num_sec_flags: int = 6,
        token_emb_dim: int = 64,
        out_dim: int = 128,
        max_degree: int = 64,
        dropout: float = 0.2
    ):
        super(NodeFeatureEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.node_type_dim = node_type_dim
        self.max_tokens = max_tokens
        self.num_sec_flags = num_sec_flags
        self.out_dim = out_dim
        
        # 1. Node type projection
        self.type_proj = nn.Linear(node_type_dim, out_dim)
        
        # 2. Token embedding table & projection
        self.token_emb = nn.Embedding(vocab_size + 20, token_emb_dim, padding_idx=0)
        self.token_proj = nn.Linear(token_emb_dim, out_dim)
        
        # 3. Structural Degree Centrality Encodings
        self.in_deg_emb = nn.Embedding(max_degree + 1, out_dim // 4)
        self.out_deg_emb = nn.Embedding(max_degree + 1, out_dim // 4)
        self.deg_proj = nn.Linear(out_dim // 2, out_dim)
        
        # 4. Security & Syntax Risk Indicator Projection
        self.sec_proj = nn.Linear(num_sec_flags, out_dim)
        
        # Normalization and fusion
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.out_linear = nn.Sequential(
            nn.Linear(out_dim * 4, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [N, node_type_dim + max_tokens + num_sec_flags]
        edge_index: [2, E]
        """
        N = x.size(0)
        
        # 1. Node type features
        if x.dim() == 1 or x.size(1) == 1:
            type_ids = x.view(-1).long().clamp(0, self.node_type_dim - 1)
            type_features = F.one_hot(type_ids, num_classes=self.node_type_dim).float()
            h_type = self.type_proj(type_features)
            h_tokens = torch.zeros_like(h_type)
            h_sec = torch.zeros_like(h_type)
        else:
            if x.size(1) >= self.node_type_dim:
                type_features = x[:, :self.node_type_dim]
            else:
                type_features = F.pad(x, (0, self.node_type_dim - x.size(1)))
            h_type = self.type_proj(type_features)
            
            # 2. Token ID features
            token_start = self.node_type_dim
            token_end = token_start + self.max_tokens
            if x.size(1) >= token_end:
                token_ids = x[:, token_start:token_end].long().clamp(0, self.vocab_size + 15)
                token_embs = self.token_emb(token_ids)
                mask = (token_ids != 0).float().unsqueeze(-1)
                h_tokens = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                h_tokens = self.token_proj(h_tokens)
            else:
                h_tokens = torch.zeros_like(h_type)
                
            # Security features
            sec_features = x[:, token_end:token_end + self.num_sec_flags] if x.size(1) >= token_end + self.num_sec_flags else torch.zeros(N, self.num_sec_flags, device=x.device)
            h_sec = self.sec_proj(sec_features)
        
        # 3. Structural Degree Centrality
        if edge_index is not None and edge_index.numel() > 0 and edge_index.size(1) > 0:
            in_deg = torch.zeros(N, dtype=torch.long, device=x.device)
            out_deg = torch.zeros(N, dtype=torch.long, device=x.device)
            ones = torch.ones(edge_index.size(1), dtype=torch.long, device=x.device)
            src = edge_index[0].clamp(0, N - 1)
            dst = edge_index[1].clamp(0, N - 1)
            out_deg.scatter_add_(0, src, ones)
            in_deg.scatter_add_(0, dst, ones)
            in_emb = self.in_deg_emb(in_deg.clamp(0, 64))
            out_emb = self.out_deg_emb(out_deg.clamp(0, 64))
            h_deg = self.deg_proj(torch.cat([in_emb, out_emb], dim=-1))
        else:
            h_deg = torch.zeros_like(h_type)
        
        combined = torch.cat([h_type, h_tokens, h_deg, h_sec], dim=-1) # [N, out_dim * 4]
        h = self.out_linear(combined)
        return self.norm(h + self.dropout(h_type))

class CodeSequenceEncoder(nn.Module):
    """
    High-Speed Multi-Scale 1D CNN + Gated Global Pooling
    on raw C/C++ source code token sequences.
    Optimized for high throughput and rich n-gram semantic extraction.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, max_seq_len: int = 128, dropout: float = 0.25):
        super(CodeSequenceEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.token_emb = nn.Embedding(vocab_size + 20, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len + 10, embed_dim)
        
        # Multi-Scale 1D Convolutions with kernel sizes (3, 5, 7) for local syntactic blocks
        self.conv3 = nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(embed_dim, embed_dim // 2, kernel_size=7, padding=3)
        self.conv_proj = nn.Linear(embed_dim * 3 // 2, embed_dim)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Fast Gated Readout
        self.gate_linear = nn.Linear(embed_dim, 1)
        
    def forward(self, seq_tokens: torch.Tensor) -> torch.Tensor:
        """
        seq_tokens: [B, L]
        returns: [B, embed_dim]
        """
        B, L = seq_tokens.size()
        pos = torch.arange(0, L, dtype=torch.long, device=seq_tokens.device).unsqueeze(0).expand(B, L)
        
        # Token + Positional Embedding: [B, L, D]
        x = self.token_emb(seq_tokens) + self.pos_emb(pos)
        x = self.dropout(x)
        
        # Multi-Scale 1D Convolutions: [B, D, L]
        x_conv = x.transpose(1, 2)
        c3 = F.gelu(self.conv3(x_conv))
        c5 = F.gelu(self.conv5(x_conv))
        c7 = F.gelu(self.conv7(x_conv))
        
        c_all = torch.cat([c3, c5, c7], dim=1).transpose(1, 2) # [B, L, 3D/2]
        c_proj = self.conv_proj(c_all) # [B, L, D]
        x_out = self.norm(x + self.dropout(c_proj))
        
        # Masked Gated Pooling Readout
        scores = self.gate_linear(x_out) # [B, L, 1]
        mask = (seq_tokens != 0).unsqueeze(-1) # [B, L, 1]
        scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1) # [B, L, 1]
        
        z_seq = (x_out * weights).sum(dim=1) # [B, D]
        return z_seq

class GlobalExpertEncoder(nn.Module):
    """Encodes high-level security indicators, cyclomatic metrics, and graph topology."""
    def __init__(self, in_dim: int = 96, out_dim: int = 128, dropout: float = 0.25):
        super(GlobalExpertEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class DualStreamGraphTransformer(nn.Module):
    """
    Advanced Multi-Modal Vulnerability Detection Architecture:
    - Stream 1: Jumping-Knowledge Multi-Head Graph Transformer on CPG (AST + CFG + DDG + Sinks)
    - Stream 2: Dilated Multi-Scale & Self-Attention Sequence Stream on Code Tokens
    - Stream 3: Global Topological & Security Expert Feature Stream
    - Fusion: Cross-Modal Bilinear & Gated Interaction Head -> Binary Classification
    """
    def __init__(
        self,
        vocab_size: int = 10000,
        node_type_dim: int = 17,
        max_tokens_per_node: int = 10,
        num_sec_flags: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        num_edge_types: int = 11,
        num_classes: int = 2,
        dropout: float = 0.3,
        drop_edge_rate: float = 0.15,
        pooling: str = "hybrid",
        max_seq_len: int = 256,
        code_vec_dim: int = 64,
        graph_vec_dim: int = 32
    ):
        super(DualStreamGraphTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_edge_rate = drop_edge_rate
        
        # 1. Graph Stream Components
        self.encoder = NodeFeatureEncoder(
            vocab_size=vocab_size,
            node_type_dim=node_type_dim,
            max_tokens=max_tokens_per_node,
            num_sec_flags=num_sec_flags,
            token_emb_dim=64,
            out_dim=hidden_dim,
            dropout=dropout
        )
        
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                num_edge_types=num_edge_types,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Jumping Knowledge (JK) Projection: Combines [h_0, h_1, ..., h_L]
        self.jk_proj = nn.Sequential(
            nn.Linear(hidden_dim * (num_layers + 1), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Multi-Scale Hybrid Graph Pooling
        self.pool = get_pooling_layer(pooling, hidden_dim)
        pool_dim = getattr(self.pool, "out_dim", hidden_dim)
        self.graph_proj = nn.Sequential(
            nn.Linear(pool_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 2. Code Sequence Stream Component (Fallback Token Encoder & Pre-trained CodeBERT)
        self.seq_encoder = CodeSequenceEncoder(
            vocab_size=vocab_size,
            embed_dim=hidden_dim,
            max_seq_len=max_seq_len,
            dropout=dropout
        )
        self.codebert_proj = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 3. Global Expert & Topological Stream
        expert_in_dim = code_vec_dim + graph_vec_dim
        self.expert_encoder = GlobalExpertEncoder(in_dim=expert_in_dim, out_dim=hidden_dim, dropout=dropout)
        
        # 4. Cross-Modal Fusion & Classification Head
        self.gate_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion vector: [gated_graph, gated_seq, abs_diff, prod, expert_repr] -> 5 * hidden_dim
        fusion_dim = hidden_dim * 5
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        seq_tokens: Optional[torch.Tensor] = None,
        code_vec: Optional[torch.Tensor] = None,
        graph_vec: Optional[torch.Tensor] = None,
        codebert_vec: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        x: [N, in_dim]
        edge_index: [2, E]
        edge_attr: [E]
        batch: [N]
        seq_tokens: [B, max_seq_len] (optional)
        code_vec: [B, 64] (optional)
        graph_vec: [B, 32] (optional)
        codebert_vec: [B, 768] (optional pre-trained embeddings)
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        
        # DropEdge during training
        if self.training and self.drop_edge_rate > 0.0 and edge_index is not None:
            edge_index, edge_attr = drop_edge(edge_index, edge_attr, drop_rate=self.drop_edge_rate, training=True)
            
        # --- STREAM 1: GRAPH TRANSFORMER WITH JUMPING KNOWLEDGE ---
        h = self.encoder(x, edge_index)
        jk_reprs = [h]
        
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
            jk_reprs.append(h)
            
        # Concatenate multi-layer representations (JK-Net)
        h_jk = self.jk_proj(torch.cat(jk_reprs, dim=-1)) # [N, hidden_dim]
        
        # Multi-scale graph readout
        z_graph_pooled = self.pool(h_jk, batch) # [num_graphs, pool_dim]
        z_graph = self.graph_proj(z_graph_pooled) # [num_graphs, hidden_dim]
        
        # --- STREAM 2: CODE REPRESENTATION STREAM (Pre-trained CodeBERT or Token Sequence) ---
        if codebert_vec is not None and codebert_vec.size(0) == num_graphs:
            if codebert_vec.dim() == 3:
                codebert_vec = codebert_vec.squeeze(1)
            z_seq = self.codebert_proj(codebert_vec) # [num_graphs, hidden_dim]
        elif seq_tokens is not None and seq_tokens.size(0) == num_graphs:
            z_seq = self.seq_encoder(seq_tokens) # [num_graphs, hidden_dim]
        else:
            z_seq = z_graph
            
        # --- STREAM 3: GLOBAL EXPERT STREAM ---
        if code_vec is not None and code_vec.size(0) == num_graphs:
            c_v = code_vec
        else:
            c_v = torch.zeros(num_graphs, 64, device=x.device)
            
        if graph_vec is not None and graph_vec.size(0) == num_graphs:
            g_v = graph_vec
        else:
            g_v = torch.zeros(num_graphs, 32, device=x.device)
            
        expert_input = torch.cat([c_v, g_v], dim=-1) # [num_graphs, 96]
        z_expert = self.expert_encoder(expert_input) # [num_graphs, hidden_dim]
        
        # --- CROSS-MODAL BILINEAR FUSION ---
        g = self.gate_layer(torch.cat([z_graph, z_seq], dim=-1)) # [num_graphs, hidden_dim]
        g_graph = g * z_graph
        g_seq = (1.0 - g) * z_seq
        
        z_diff = torch.abs(z_graph - z_seq)
        z_prod = z_graph * z_seq
        
        z_fused = torch.cat([g_graph, g_seq, z_diff, z_prod, z_expert], dim=-1) # [num_graphs, hidden_dim * 5]
        
        logits = self.classifier(z_fused)
        return logits

# Alias for backward compatibility
GraphTransformer = DualStreamGraphTransformer
