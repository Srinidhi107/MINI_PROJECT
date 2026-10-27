import os
import glob
import json
import re
import pandas as pd
import numpy as np
import torch
from typing import List, Optional, Dict

try:
    from torch_geometric.data import Dataset, Data
except ImportError:
    from torch.utils.data import Dataset
    Data = object

from src.graph_features import tokenize_code

EXPERT_PATTERNS = [
    r'\bmemcpy\b', r'\bstrcpy\b', r'\bstrncpy\b', r'\bstrcat\b', r'\bsprintf\b', r'\bsnprintf\b',
    r'\bmalloc\b', r'\bfree\b', r'\bcalloc\b', r'\brealloc\b', r'\bmemmove\b', r'\bmemset\b',
    r'\bsizeof\b', r'\bassert\b', r'\bgoto\b', r'\bNULL\b', r'->', r'\[\s*\w+\s*\+\s*\w+\s*\]',
    r'(\w+)\s*==\s*NULL', r'(\w+)\s*!=\s*NULL', r'<\s*0', r'<=\s*0', r'==\s*-1', r'!=\s*-1',
    r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\bswitch\b', r'\bcase\b', r'\breturn\b',
    r'\bint\b', r'\bchar\b', r'\bvoid\b', r'\buint8_t\b', r'\buint16_t\b', r'\buint32_t\b', r'\buint64_t\b',
    r'\bunsigned\b', r'\blong\b', r'\bshort\b', r'\bfloat\b', r'\bdouble\b', r'\bstruct\b',
    r'\bstatic\b', r'\bconst\b', r'\bvolatile\b', r'\btypedef\b', r'\benum\b',
    r'\*', r'&', r'<<', r'>>', r'\^', r'\|', r'%', r'/', r'\+\+', r'--', r'!=', r'==', r'<=', r'>='
]

COMPILED_PATTERNS = [re.compile(pat, re.IGNORECASE) for pat in EXPERT_PATTERNS]
DECISION_PAT = re.compile(r'\b(if|for|while|case)\b|&&|\|\||\?')
POINTER_PAT = re.compile(r'->|\*\w+')
MEM_PAT = re.compile(r'\b(malloc|free|calloc|realloc|memcpy|memset|memmove|strcpy|strncpy)\b', re.IGNORECASE)

def extract_code_vector(code_str: str) -> List[float]:
    """Extract normalized expert security & syntactic features (64 dims)."""
    s = str(code_str) if code_str else ""
    tokens = tokenize_code(s)
    num_tokens = max(len(tokens), 1)
    
    vec = [
        np.log1p(len(s)),
        np.log1p(len(s.splitlines())),
        np.log1p(num_tokens)
    ]
    
    for pat in COMPILED_PATTERNS:
        cnt = len(pat.findall(s))
        vec.append(np.log1p(cnt))
        
    decisions = len(DECISION_PAT.findall(s))
    vec.append(np.log1p(decisions))
    
    pointer_ops = len(POINTER_PAT.findall(s))
    vec.append(pointer_ops / float(num_tokens))
    
    mem_ops = len(MEM_PAT.findall(s))
    vec.append(np.log1p(mem_ops))
    
    while len(vec) < 64:
        vec.append(0.0)
    return vec[:64]

class DevignGraphDataset(Dataset):
    """
    PyTorch / PyG Multi-Modal Dataset:
    1. Precomputed CPG graph (x, edge_index, edge_attr, graph_vec)
    2. Function code token sequences (seq_tokens [1, max_seq_len])
    3. Expert security and complexity vectors (code_vec [1, 64])
    Fast precomputed in-memory caching.
    """
    def __init__(
        self,
        root_dir: str = "data/graphs",
        split: str = "train",
        vocab_path: str = "data/graphs/vocab.json",
        max_seq_len: int = 256,
        preload: bool = False,
        transform=None,
        pre_transform=None
    ):
        self.root_dir = root_dir
        self.split = split
        self.max_seq_len = max_seq_len
        self.transform = transform
        self.pre_transform = pre_transform
        
        # Load vocab
        self.vocab = {}
        if os.path.exists(vocab_path):
            try:
                with open(vocab_path, "r", encoding="utf-8") as f:
                    self.vocab = json.load(f)
            except Exception:
                pass
                
        self.unk_id = self.vocab.get("<unk>", 1)
        self.pad_id = self.vocab.get("<pad>", 0)
        
        self.split_dir = os.path.join(root_dir, split)
        self.file_list = sorted(glob.glob(os.path.join(self.split_dir, "*.pt")))
        self.cache = {}
        
        # Default fallback tensors
        self.default_seq = torch.zeros((1, self.max_seq_len), dtype=torch.long)
        self.default_code = torch.zeros((1, 64), dtype=torch.float32)
        self.default_graph = torch.zeros((1, 32), dtype=torch.float32)
        
        # Load CodeBERT embeddings if present
        self.codebert_dict = None
        cb_path = os.path.join(root_dir, "codebert_embeddings.pt")
        if os.path.exists(cb_path):
            try:
                self.codebert_dict = torch.load(cb_path, map_location="cpu", weights_only=False)
            except Exception:
                self.codebert_dict = None

        if preload and len(self.file_list) > 0:
            for idx in range(len(self.file_list)):
                self.get(idx)

    def _attach_features(self, data):
        """Fast O(1) dictionary attachment."""
        if self.codebert_dict is not None:
            f_id = str(getattr(data, "func_id", ""))
            if f_id in self.codebert_dict:
                data.codebert_vec = self.codebert_dict[f_id]
            else:
                data.codebert_vec = torch.zeros((1, 768), dtype=torch.float32)

        if hasattr(data, "code_vec") and data.code_vec is not None:
            return
            
        func_id = getattr(data, "func_id", "")
        if hasattr(self, "feature_lookup") and str(func_id) in self.feature_lookup:
            seq_t, code_t = self.feature_lookup[str(func_id)]
            data.seq_tokens = seq_t
            data.code_vec = code_t
        else:
            data.seq_tokens = self.default_seq
            data.code_vec = self.default_code
            
        if not hasattr(data, "graph_vec") or data.graph_vec is None:
            data.graph_vec = self.default_graph

    def len(self) -> int:
        return len(self.file_list)
        
    def __len__(self) -> int:
        return len(self.file_list)
        
    def get(self, idx: int):
        if idx in self.cache:
            return self.cache[idx]
        filepath = self.file_list[idx]
        data = torch.load(filepath, weights_only=False)
        self._attach_features(data)
        self.cache[idx] = data
        return data
        
    def __getitem__(self, idx: int):
        return self.get(idx)
        
    def get_class_counts(self):
        """Count class distribution across dataset."""
        counts = {0: 0, 1: 0}
        for i in range(len(self.file_list)):
            d = self.get(i)
            label = int(d.y.item()) if hasattr(d, 'y') else 0
            counts[label] = counts.get(label, 0) + 1
        return counts
