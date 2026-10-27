import re
import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

# Standard Joern Node Types
NODE_TYPES = [
    "METHOD", "METHOD_PARAMETER_IN", "METHOD_PARAMETER_OUT", "METHOD_RETURN",
    "BLOCK", "CALL", "IDENTIFIER", "LITERAL", "RETURN", "CONTROL_STRUCTURE",
    "LOCAL", "TYPE_DECL", "TYPE_REF", "BINDING", "FIELD_IDENTIFIER",
    "JUMP_TARGET", "UNKNOWN"
]
NODE_TYPE_TO_ID = {nt: i for i, nt in enumerate(NODE_TYPES)}

# Standard Joern Edge Types (AST, CFG, PDG/Data-Dep, etc.)
EDGE_TYPES = [
    "AST", "CFG", "CDG", "DDG", "REACHING_DEF", "ARGUMENT", "REF", "CALL", "BINDS", "OTHER"
]
EDGE_TYPE_TO_ID = {et: i for i, et in enumerate(EDGE_TYPES)}

# Security Vulnerability Patterns for Node & Graph Features
SINK_PATTERNS = re.compile(
    r'\b(memcpy|strcpy|strncpy|strcat|strncat|sprintf|snprintf|vsprintf|vsnprintf|'
    r'gets|scanf|sscanf|fscanf|malloc|calloc|realloc|free|memmove|memset|bcopy|'
    r'alloca|read|recv|recvfrom|system|popen|exec|execl|execv|execvp)\b',
    re.IGNORECASE
)

CONTROL_PATTERNS = re.compile(r'\b(if|else|while|for|switch|case|default|return|goto|break|continue)\b')
POINTER_PATTERNS = re.compile(r'->|\*\w+|\&\w+|\[\s*\w+\s*\]')
ARITH_PATTERNS = re.compile(r'[+\-*/%&|^<<>>]+|\+\+|--')
CHECK_PATTERNS = re.compile(r'==|!=|<=|>=|<|>|\bNULL\b|\bnull\b|\bnullptr\b|-1')

def tokenize_code(code_str: str) -> List[str]:
    """Tokenize C/C++ code snippet into words and symbols."""
    if not code_str or code_str == "<empty>":
        return []
    tokens = re.findall(r'[A-Za-z_][A-Za-z0-9_]*|\d+|[+\-*/%&|^!=<>]+|[{}()\[\];,.]', str(code_str))
    return [t.lower() for t in tokens]

class CPGParser:
    def __init__(self, token_vocab: Optional[Dict[str, int]] = None, max_tokens_per_node: int = 10):
        self.node_type_map = NODE_TYPE_TO_ID
        self.edge_type_map = EDGE_TYPE_TO_ID
        self.token_vocab = token_vocab or {}
        self.max_tokens_per_node = max_tokens_per_node
        
    def parse_dot_file(self, dot_filepath: str) -> Tuple[List[Dict], List[Tuple[str, str, str]]]:
        """
        Robustly parses a Joern exported .dot file into node list and edge list,
        handling multi-line code attributes and complex DOT structures.
        """
        nodes = []
        edges = []
        
        with open(dot_filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            
        # Match node definitions: "id" [label="LABEL" ... ];
        # Using DOTALL so multi-line attributes inside brackets are fully captured
        node_pattern = re.compile(r'"(\d+)"\s*\[label="([^"]+)"(.*?)\];', re.DOTALL)
        edge_pattern = re.compile(r'"(\d+)"\s*->\s*"(\d+)"\s*\[label="([^"]+)"', re.DOTALL)
        code_pattern = re.compile(r'CODE="((?:[^"\\]|\\.)*)"', re.DOTALL)
        
        for match in node_pattern.finditer(content):
            node_id, label, rest = match.groups()
            code_match = code_pattern.search(rest)
            code_text = code_match.group(1) if code_match else ""
            
            nodes.append({
                "id": node_id,
                "label": label.upper(),
                "code": code_text
            })
            
        for match in edge_pattern.finditer(content):
            src, dst, elabel = match.groups()
            edges.append((src, dst, elabel.upper()))
            
        return nodes, edges

    def build_node_features(self, nodes: List[Dict], vocab: Dict[str, int]) -> torch.Tensor:
        """
        Encode:
        1. Node type one-hot (17 dims)
        2. Token IDs for node code snippet (max_tokens_per_node dims)
        3. Node-level security & structural flags (6 dims: is_sink, is_control, is_pointer, is_arith, is_check, code_len)
        Total dim = 17 + max_tokens_per_node + 6
        """
        features = []
        pad_id = vocab.get("<pad>", 0)
        unk_id = vocab.get("<unk>", 1)
        
        for n in nodes:
            # 1. Node type one-hot
            type_idx = self.node_type_map.get(n["label"], self.node_type_map["UNKNOWN"])
            type_vec = [0.0] * len(NODE_TYPES)
            type_vec[type_idx] = 1.0
            
            # 2. Token IDs
            code_txt = n["code"]
            tokens = tokenize_code(code_txt)
            token_ids = [vocab.get(t, unk_id) for t in tokens[:self.max_tokens_per_node]]
            while len(token_ids) < self.max_tokens_per_node:
                token_ids.append(pad_id)
                
            # 3. Security & Semantic flags
            is_sink = 1.0 if SINK_PATTERNS.search(code_txt) else 0.0
            is_control = 1.0 if CONTROL_PATTERNS.search(code_txt) else 0.0
            is_pointer = 1.0 if POINTER_PATTERNS.search(code_txt) else 0.0
            is_arith = 1.0 if ARITH_PATTERNS.search(code_txt) else 0.0
            is_check = 1.0 if CHECK_PATTERNS.search(code_txt) else 0.0
            code_len_norm = min(float(len(code_txt)) / 100.0, 5.0)
            
            sem_vec = [is_sink, is_control, is_pointer, is_arith, is_check, code_len_norm]
            
            combined = type_vec + [float(tid) for tid in token_ids] + sem_vec
            features.append(combined)
            
        feature_dim = len(NODE_TYPES) + self.max_tokens_per_node + 6
        if not features:
            return torch.zeros((0, feature_dim), dtype=torch.float32)
            
        return torch.tensor(features, dtype=torch.float32)

    def compute_graph_vector(self, nodes: List[Dict], edges: List[Tuple[str, str, str]], num_nodes: int) -> torch.Tensor:
        """
        Compute global topological graph descriptors (32 dims):
        - Counts of node types (17 dims)
        - Counts of edge types (10 dims)
        - Security metrics (sink count, control count, pointer op count) (3 dims)
        - Graph density & edge-to-node ratio (2 dims)
        """
        # Node type frequencies
        node_counts = [0.0] * len(NODE_TYPES)
        sink_count = 0.0
        control_count = 0.0
        pointer_count = 0.0
        
        for n in nodes:
            t_idx = self.node_type_map.get(n["label"], self.node_type_map["UNKNOWN"])
            node_counts[t_idx] += 1.0
            c = n["code"]
            if SINK_PATTERNS.search(c):
                sink_count += 1.0
            if CONTROL_PATTERNS.search(c):
                control_count += 1.0
            if POINTER_PATTERNS.search(c):
                pointer_count += 1.0
                
        # Edge type frequencies
        edge_counts = [0.0] * len(EDGE_TYPES)
        for _, _, etype in edges:
            e_idx = self.edge_type_map.get(etype, self.edge_type_map["OTHER"])
            edge_counts[e_idx] += 1.0
            
        num_e = float(len(edges))
        num_n = max(float(num_nodes), 1.0)
        density = num_e / (num_n * (num_n - 1) / 2.0) if num_n > 1 else 0.0
        ratio = num_e / num_n
        
        # Normalize counts by log1p
        norm_node_counts = [np.log1p(c) for c in node_counts]
        norm_edge_counts = [np.log1p(c) for c in edge_counts]
        norm_sec = [np.log1p(sink_count), np.log1p(control_count), np.log1p(pointer_count)]
        
        vec = norm_node_counts + norm_edge_counts + norm_sec + [density, ratio]
        return torch.tensor(vec, dtype=torch.float32)

    def convert_to_pyg_data(self, dot_filepath: str, target: int, func_id: str, vocab: Dict[str, int]):
        """Convert a DOT CPG file into PyTorch Geometric Data format."""
        try:
            from torch_geometric.data import Data
        except ImportError:
            class Data:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)
                        
        nodes, raw_edges = self.parse_dot_file(dot_filepath)
        
        if not nodes:
            return None
            
        # Map node string IDs to sequential indices 0..N-1
        id_to_idx = {n["id"]: idx for idx, n in enumerate(nodes)}
        
        # Build node features
        x = self.build_node_features(nodes, vocab)
        
        # Build edge_index and edge_attr
        src_indices = []
        dst_indices = []
        edge_attrs = []
        
        for src, dst, edge_type in raw_edges:
            if src in id_to_idx and dst in id_to_idx:
                src_indices.append(id_to_idx[src])
                dst_indices.append(id_to_idx[dst])
                edge_type_id = self.edge_type_map.get(edge_type, self.edge_type_map["OTHER"])
                edge_attrs.append(edge_type_id)
                
        if src_indices:
            edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0,), dtype=torch.long)
            
        graph_vec = self.compute_graph_vector(nodes, raw_edges, len(nodes))
        y = torch.tensor([target], dtype=torch.long)
        
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            graph_vec=graph_vec.unsqueeze(0), # [1, 32]
            y=y,
            num_nodes=len(nodes),
            func_id=func_id
        )
        return data
