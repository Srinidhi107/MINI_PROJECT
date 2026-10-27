import os
import sys
import re
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from torch_geometric.data import Data
except ImportError:
    Data = object

from src.dataset import extract_code_vector
from src.graph_features import tokenize_code

def main():
    src_dir = "devign_pyg_10000"
    if not os.path.exists(src_dir):
        print(f"Error: {src_dir} directory not found.")
        return

    # Create destination directories
    train_dir = "data/graphs/train"
    val_dir = "data/graphs/val"
    test_dir = "data/graphs/test"
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    # Load vocab
    vocab_path = "data/graphs/vocab.json"
    vocab = {}
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
    unk_id = vocab.get("<unk>", 1)
    pad_id = vocab.get("<pad>", 0)
    max_seq_len = 128

    # Load CSV data
    print("Loading CSV datasets for feature lookup...", flush=True)
    df_train = pd.read_csv("data/processed/train.csv") if os.path.exists("data/processed/train.csv") else pd.DataFrame()
    df_val = pd.read_csv("data/processed/validation.csv") if os.path.exists("data/processed/validation.csv") else pd.DataFrame()
    df_test = pd.read_csv("data/processed/test.csv") if os.path.exists("data/processed/test.csv") else pd.DataFrame()
    
    full_df = pd.concat([df_train, df_val, df_test], ignore_index=True) if len(df_train) > 0 else pd.DataFrame()
    id_to_row = {}
    if len(full_df) > 0 and "id" in full_df.columns:
        for idx, row in full_df.iterrows():
            id_to_row[int(row["id"])] = row

    files = sorted(os.listdir(src_dir))
    print(f"Found {len(files)} total files in {src_dir}", flush=True)

    counts = {"train": 0, "val": 0, "test": 0}
    pos_counts = {"train": 0, "val": 0, "test": 0}

    for fname in tqdm(files, desc="Ingesting Graphs"):
        src_path = os.path.join(src_dir, fname)
        if os.path.getsize(src_path) == 0:
            continue

        try:
            raw = torch.load(src_path, map_location="cpu", weights_only=False)
        except Exception:
            continue

        # Determine split
        if fname.startswith("test"):
            split = "test"
            dest_folder = test_dir
            df_ref = df_test
        elif fname.startswith("val"):
            split = "val"
            dest_folder = val_dir
            df_ref = df_val
        elif fname.startswith("train"):
            split = "train"
            dest_folder = train_dir
            df_ref = df_train
        else:
            continue

        # Extract graph attributes
        if isinstance(raw, dict):
            x = raw.get("x")
            edge_index = raw.get("edge_index")
            edge_attr = raw.get("edge_type", raw.get("edge_attr"))
            y = raw.get("y")
        else:
            x = getattr(raw, "x", None)
            edge_index = getattr(raw, "edge_index", None)
            edge_attr = getattr(raw, "edge_type", getattr(raw, "edge_attr", None))
            y = getattr(raw, "y", None)

        if x is None or edge_index is None:
            continue

        # Resolve ground truth label y
        label = None
        if y is not None:
            label = int(y.item() if hasattr(y, "item") else y[0])
        else:
            m = re.match(r"[a-z]+_(\d+)-cpg\.pt", fname)
            if m:
                idx_num = int(m.group(1))
                if idx_num in id_to_row:
                    row = id_to_row[idx_num]
                    t_val = row.get("target", row.get("label", None))
                    if t_val is not None:
                        label = int(bool(t_val))
                elif idx_num < len(df_ref):
                    row = df_ref.iloc[idx_num]
                    t_val = row.get("target", row.get("label", None))
                    if t_val is not None:
                        label = int(bool(t_val))

        if label is None:
            continue

        y_tensor = torch.tensor([label], dtype=torch.long)
        if edge_attr is None:
            edge_attr = torch.zeros(edge_index.size(1), dtype=torch.long)
        else:
            edge_attr = edge_attr.long()

        # Format x
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Code features & sequences
        code_text = ""
        m = re.match(r"[a-z]+_(\d+)-cpg\.pt", fname)
        if m:
            idx_num = int(m.group(1))
            if idx_num in id_to_row:
                row = id_to_row[idx_num]
                code_text = str(row.get("func_clean", row.get("func", "")))
            elif idx_num < len(df_ref):
                row = df_ref.iloc[idx_num]
                code_text = str(row.get("func_clean", row.get("func", "")))

        if code_text and code_text != "nan":
            tokens = tokenize_code(code_text)
            t_ids = [vocab.get(t, unk_id) for t in tokens[:max_seq_len]]
            while len(t_ids) < max_seq_len:
                t_ids.append(pad_id)
            seq_tokens = torch.tensor([t_ids], dtype=torch.long)
            c_vec = extract_code_vector(code_text)
            code_vec = torch.tensor([c_vec], dtype=torch.float32)
        else:
            seq_tokens = torch.zeros((1, max_seq_len), dtype=torch.long)
            code_vec = torch.zeros((1, 64), dtype=torch.float32)

        # Build PyG Data object
        data_obj = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y_tensor,
            seq_tokens=seq_tokens,
            code_vec=code_vec,
            graph_vec=torch.zeros((1, 32), dtype=torch.float32),
            func_id=fname.replace("-cpg.pt", "")
        )

        out_path = os.path.join(dest_folder, fname)
        torch.save(data_obj, out_path)
        counts[split] += 1
        pos_counts[split] += label

    print("\n" + "=" * 60)
    print("Ingestion Completed Successfully!")
    print("=" * 60)
    for sp in ["train", "val", "test"]:
        tot = counts[sp]
        pos = pos_counts[sp]
        pct = (pos / tot * 100) if tot > 0 else 0
        print(f"Split [{sp:5s}]: {tot:5d} graphs (Vulnerable: {pos:4d} [{pct:.1f}%], Safe: {tot-pos:4d})")

if __name__ == "__main__":
    main()
