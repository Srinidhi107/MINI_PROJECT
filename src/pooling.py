import torch
import torch.nn as nn

class GlobalMeanPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GlobalMeanPool, self).__init__()
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, hidden_dim]
        batch: [num_nodes] batch index tensor
        """
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        out = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
        counts = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
        
        out.index_add_(0, batch, x)
        counts.index_add_(0, batch, torch.ones(batch.size(0), 1, device=x.device, dtype=x.dtype))
        
        return out / counts.clamp(min=1e-9)

class GlobalMaxPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GlobalMaxPool, self).__init__()
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        out = torch.full((num_graphs, x.size(1)), -1e9, device=x.device, dtype=x.dtype)
        
        # Vectorized max scatter
        batch_expanded = batch.unsqueeze(1).expand(-1, x.size(1))
        out.scatter_reduce_(0, batch_expanded, x, reduce="amax", include_self=False)
        # Replace any untouched with zero
        out = torch.where(out == -1e9, torch.zeros_like(out), out)
        return out

class GlobalAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super(GlobalAttentionPool, self).__init__()
        self.out_dim = hidden_dim
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        gate_scores = self.gate_nn(x) # [num_nodes, 1]
        
        # Per-graph numerical stable softmax via scatter_reduce
        max_scores = torch.full((num_graphs, 1), -1e9, device=x.device, dtype=x.dtype)
        max_scores.scatter_reduce_(0, batch.unsqueeze(1), gate_scores, reduce="amax", include_self=False)
        
        exp_scores = torch.exp(gate_scores - max_scores[batch])
        sum_exp = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
        sum_exp.index_add_(0, batch, exp_scores)
        
        weights = exp_scores / sum_exp[batch].clamp(min=1e-9) # [num_nodes, 1]
        
        weighted_x = x * weights # [num_nodes, hidden_dim]
        out = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
        out.index_add_(0, batch, weighted_x)
        return out

class MultiHeadAttentionPool(nn.Module):
    """
    Multi-Head Attention Readout (Pooling by Multihead Attention / Prototype Queries):
    Uses multiple attention heads where each head specializes in learning different
    vulnerability patterns (e.g., memory management, buffer bounds, pointer arithmetic).
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super(MultiHeadAttentionPool, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_heads)
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        scores = self.gate_nn(x)  # [num_nodes, num_heads]
        
        # Per-graph numerical stable softmax across all heads
        max_scores = torch.full((num_graphs, self.num_heads), -1e9, device=x.device, dtype=x.dtype)
        max_scores.scatter_reduce_(0, batch.unsqueeze(1).expand(-1, self.num_heads), scores, reduce="amax", include_self=False)
        
        exp_scores = torch.exp(scores - max_scores[batch])
        sum_exp = torch.zeros(num_graphs, self.num_heads, device=x.device, dtype=x.dtype)
        sum_exp.index_add_(0, batch, exp_scores)
        
        weights = exp_scores / sum_exp[batch].clamp(min=1e-9)  # [num_nodes, num_heads]
        
        # Weighted pooling per head
        pooled_heads = []
        for h in range(self.num_heads):
            w_h = weights[:, h:h+1]  # [num_nodes, 1]
            head_out = torch.zeros(num_graphs, self.hidden_dim, device=x.device, dtype=x.dtype)
            head_out.index_add_(0, batch, x * w_h)
            pooled_heads.append(head_out)
            
        # Combine heads via mean across heads then projection
        combined = torch.stack(pooled_heads, dim=0).mean(dim=0)  # [num_graphs, hidden_dim]
        return self.proj(combined)

class TopKAttentionPool(nn.Module):
    """
    Top-K Salient Node Readout:
    Filters out boilerplate nodes (e.g., trivial declarations, empty returns)
    and pools only the top-K highest scoring statements where vulnerability signals concentrate.
    """
    def __init__(self, hidden_dim: int, top_k_ratio: float = 0.25, min_k: int = 5):
        super(TopKAttentionPool, self).__init__()
        self.hidden_dim = hidden_dim
        self.top_k_ratio = top_k_ratio
        self.min_k = min_k
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.out_dim = hidden_dim * 2  # Concat of top-k mean + top-k max

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        scores = self.score_net(x).squeeze(-1)  # [num_nodes]
        
        means = []
        maxs = []
        for g in range(num_graphs):
            mask = (batch == g)
            if not mask.any():
                means.append(torch.zeros(self.hidden_dim, device=x.device))
                maxs.append(torch.zeros(self.hidden_dim, device=x.device))
                continue
                
            g_x = x[mask]
            g_scores = scores[mask]
            k = max(self.min_k, int(g_x.size(0) * self.top_k_ratio))
            k = min(k, g_x.size(0))
            
            topk_indices = torch.topk(g_scores, k=k, largest=True).indices
            topk_x = g_x[topk_indices]
            
            means.append(topk_x.mean(dim=0))
            maxs.append(topk_x.max(dim=0).values)
            
        h_mean = torch.stack(means, dim=0)
        h_max = torch.stack(maxs, dim=0)
        return torch.cat([h_mean, h_max], dim=-1)

class VulnerabilityFocusedPool(nn.Module):
    """
    Vulnerability-Focused Readout for Devign:
    Combines Multi-Head Attention Readout (diverse vulnerability scopes) with
    Extreme Feature Max Readout (existential danger signals) and Top-Salient Readout.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super(VulnerabilityFocusedPool, self).__init__()
        self.mha_pool = MultiHeadAttentionPool(hidden_dim, num_heads=num_heads)
        self.max_pool = GlobalMaxPool(hidden_dim)
        self.out_dim = hidden_dim * 2
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h_mha = self.mha_pool(x, batch)
        h_max = self.max_pool(x, batch)
        return self.fuse(torch.cat([h_mha, h_max], dim=-1))

class GlobalHybridPool(nn.Module):
    """
    Multi-Scale Graph Readout:
    Concatenates Mean Pool (Global Context) + Max Pool (Salient Extreme Nodes) + Attention Pool (Vulnerability Focus).
    """
    def __init__(self, hidden_dim: int):
        super(GlobalHybridPool, self).__init__()
        self.mean_pool = GlobalMeanPool(hidden_dim)
        self.max_pool = GlobalMaxPool(hidden_dim)
        self.attn_pool = GlobalAttentionPool(hidden_dim)
        self.out_dim = hidden_dim * 3
        
    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        h_mean = self.mean_pool(x, batch)
        h_max = self.max_pool(x, batch)
        h_attn = self.attn_pool(x, batch)
        return torch.cat([h_mean, h_max, h_attn], dim=-1)

def get_pooling_layer(pool_type: str, hidden_dim: int) -> nn.Module:
    p_lower = str(pool_type).lower()
    if p_lower in ["vuln", "vulnerability", "vulnerability_focused", "vuln_focused"]:
        return VulnerabilityFocusedPool(hidden_dim, num_heads=4)
    elif p_lower in ["multihead", "mha", "multi_head", "multihead_attn"]:
        return MultiHeadAttentionPool(hidden_dim, num_heads=4)
    elif p_lower in ["topk", "top_k", "salient"]:
        return TopKAttentionPool(hidden_dim, top_k_ratio=0.25)
    elif p_lower in ["hybrid", "multiscale", "multi_scale", "concat"]:
        return GlobalHybridPool(hidden_dim)
    elif p_lower in ["attn", "attention", "gated"]:
        return GlobalAttentionPool(hidden_dim)
    elif p_lower == "max":
        return GlobalMaxPool(hidden_dim)
    else:
        return GlobalMeanPool(hidden_dim)

