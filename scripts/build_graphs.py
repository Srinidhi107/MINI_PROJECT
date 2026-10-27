import os
import sys
import glob
import json
import argparse
from collections import Counter
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.graph_features import CPGParser, tokenize_code

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

def build_vocabulary(dot_files: list, max_vocab_size: int = 10000) -> dict:
    """Build token vocabulary from code snippets in DOT files."""
    print(f"\nBuilding token vocabulary from {len(dot_files)} CPG DOT files...", flush=True)
    token_counts = Counter()
    parser = CPGParser(max_tokens_per_node=10)
    
    for dot_file in tqdm(dot_files, desc="Scanning tokens"):
        nodes, _ = parser.parse_dot_file(dot_file)
        for n in nodes:
            tokens = tokenize_code(n.get("code", ""))
            token_counts.update(tokens)
            
    # Special tokens
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, _ in token_counts.most_common(max_vocab_size - 2):
        if token not in vocab:
            vocab[token] = len(vocab)
            
    print(f"Vocabulary constructed: {len(vocab):,} unique tokens (including <pad>, <unk>).", flush=True)
    return vocab

def build_graphs_for_split(split_name: str, vocab: dict):
    dot_dir = os.path.join(PROJECT_ROOT, "data", "graphs", "raw_dot", split_name)
    output_graph_dir = os.path.join(PROJECT_ROOT, "data", "graphs", split_name)
    manifest_path = os.path.join(PROJECT_ROOT, "data", "raw", "devign", split_name, f"manifest_{split_name}.csv")
    
    os.makedirs(output_graph_dir, exist_ok=True)
    
    # Load manifest for target labels
    manifest = {}
    if os.path.exists(manifest_path):
        m_df = pd.read_csv(manifest_path)
        for _, row in m_df.iterrows():
            manifest[str(row['function_id'])] = int(row['target'])
    else:
        # Fallback to train_clean.csv if manifest not found
        clean_csv = os.path.join(PROJECT_ROOT, "data", "processed", f"{split_name}_clean.csv")
        if os.path.exists(clean_csv):
            m_df = pd.read_csv(clean_csv)
            for _, row in m_df.iterrows():
                manifest[str(row['function_id'])] = int(row['target'])
                
    dot_files = glob.glob(os.path.join(dot_dir, "*.dot"))
    if not dot_files:
        print(f"No .dot files found in {dot_dir}", flush=True)
        return
        
    print(f"\nConverting {len(dot_files)} DOT files to PyG Data objects for split '{split_name}'...", flush=True)
    
    parser = CPGParser(token_vocab=vocab, max_tokens_per_node=10)
    saved = 0
    skipped = 0
    
    for dot_file in tqdm(dot_files, desc=f"Building PyG [{split_name}]"):
        func_id = os.path.splitext(os.path.basename(dot_file))[0]
        target = manifest.get(func_id, 0)
        
        pyg_data = parser.convert_to_pyg_data(dot_file, target=target, func_id=func_id, vocab=vocab)
        if pyg_data is not None and pyg_data.num_nodes > 0:
            out_path = os.path.join(output_graph_dir, f"{func_id}.pt")
            torch.save(pyg_data, out_path)
            saved += 1
        else:
            skipped += 1
            
    print(f"Split '{split_name}': {saved} graphs saved to '{output_graph_dir}', {skipped} skipped.", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Build PyTorch Geometric graph data from Joern DOT files.")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--vocab_size", type=int, default=10000, help="Maximum vocabulary size")
    parser.add_argument("--rebuild_vocab", action="store_true", help="Force rebuild of vocabulary file")
    
    args = parser.parse_args()
    
    print("=" * 70, flush=True)
    print("STEP 5: GRAPH CONSTRUCTION (PYTORCH GEOMETRIC - ENHANCED)", flush=True)
    print("=" * 70, flush=True)
    
    raw_dot_root = os.path.join(PROJECT_ROOT, "data", "graphs", "raw_dot")
    all_dot_files = glob.glob(os.path.join(raw_dot_root, "*", "*.dot"))
    
    if not all_dot_files:
        print(f"Error: No DOT files found under {raw_dot_root}. Please run generate_cpg.py first.", flush=True)
        return
        
    vocab_path = os.path.join(PROJECT_ROOT, "data", "graphs", "vocab.json")
    if os.path.exists(vocab_path) and not args.rebuild_vocab:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"Loaded existing vocabulary ({len(vocab)} tokens) from '{vocab_path}'", flush=True)
    else:
        vocab = build_vocabulary(all_dot_files, max_vocab_size=args.vocab_size)
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, indent=2)
        print(f"Saved vocabulary to '{vocab_path}'", flush=True)
        
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for split in splits:
        build_graphs_for_split(split, vocab=vocab)
        
    print("\n" + "=" * 70, flush=True)
    print("Enhanced Graph construction completed successfully!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
