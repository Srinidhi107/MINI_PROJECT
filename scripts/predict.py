import os
import sys
import json
import argparse
import tempfile
import shutil
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.graph_features import CPGParser, tokenize_code
from src.dataset import extract_code_vector
from src.graph_transformer import GraphTransformer
from src.utils import load_config
from scripts.generate_cpg import generate_cpg_for_file

def load_trained_model(checkpoint_path: str = "models/checkpoints/best_graph_transformer.pt", config_path: str = "configs/config.yaml"):
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device") == "cuda" else "cpu")
    
    vocab_path = config["data"]["vocab_path"]
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
        
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    m_cfg = config["model"]
    model = GraphTransformer(
        vocab_size=len(vocab),
        node_type_dim=config["data"].get("node_type_dim", 17),
        max_tokens_per_node=config["data"].get("max_tokens_per_node", 10),
        num_sec_flags=6,
        hidden_dim=m_cfg["hidden_dim"],
        num_layers=m_cfg["num_layers"],
        num_heads=m_cfg["num_heads"],
        num_edge_types=config["data"]["edge_types"],
        num_classes=config["data"]["num_classes"],
        dropout=0.0, # Evaluation mode
        drop_edge_rate=0.0,
        pooling=m_cfg["pooling"]
    ).to(device)
    
    threshold = 0.50
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        threshold = float(checkpoint.get("optimal_threshold", 0.50))
    else:
        print(f"[WARNING] Checkpoint not found at '{checkpoint_path}'. Using initialized model.")
        
    model.eval()
    return model, vocab, device, threshold

def predict_c_code(code_str: str, model, vocab, device, threshold: float = 0.50, verbose: bool = True) -> dict:
    """End-to-end inference from raw C source code to vulnerability prediction."""
    temp_dir = tempfile.mkdtemp(prefix="devign_pred_")
    c_filepath = os.path.join(temp_dir, "input_func.c")
    dot_filepath = os.path.join(temp_dir, "input_func.dot")
    
    with open(c_filepath, "w", encoding="utf-8") as f:
        f.write(code_str)
        
    try:
        # Step 1: Generate CPG with Joern
        success = generate_cpg_for_file(c_filepath, dot_filepath, temp_dir)
        if not success or not os.path.exists(dot_filepath):
            return {
                "status": "error",
                "message": "Joern CPG extraction failed. Ensure valid C function syntax."
            }
            
        # Step 2: Parse CPG into PyG Data Object
        parser = CPGParser(token_vocab=vocab, max_tokens_per_node=10)
        nodes, edges = parser.parse_dot_file(dot_filepath)
        pyg_data = parser.convert_to_pyg_data(dot_filepath, target=0, func_id="query", vocab=vocab)
        
        if pyg_data is None or pyg_data.num_nodes == 0:
            return {
                "status": "error",
                "message": "Failed to construct graph representation from CPG."
            }
            
        # Step 3: Extract Sequence & Expert Features
        tokens = tokenize_code(code_str)
        unk_id = vocab.get("<unk>", 1)
        pad_id = vocab.get("<pad>", 0)
        token_ids = [vocab.get(t, unk_id) for t in tokens[:256]]
        while len(token_ids) < 256:
            token_ids.append(pad_id)
        seq_tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
        
        code_vec_raw = extract_code_vector(code_str)
        code_vec = torch.tensor([code_vec_raw], dtype=torch.float32, device=device)
        graph_vec = pyg_data.graph_vec.to(device) if hasattr(pyg_data, 'graph_vec') and pyg_data.graph_vec is not None else torch.zeros((1, 32), device=device)
        
        pyg_data = pyg_data.to(device)
        with torch.no_grad():
            logits = model(
                pyg_data.x,
                pyg_data.edge_index,
                pyg_data.edge_attr,
                batch=None,
                seq_tokens=seq_tokens,
                code_vec=code_vec,
                graph_vec=graph_vec
            )
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
            
        safe_prob = float(probs[0])
        vuln_prob = float(probs[1])
        pred_label = 1 if vuln_prob >= threshold else 0
        
        if vuln_prob < 0.40:
            risk_tier = "SAFE (LOW RISK)"
        elif vuln_prob < 0.60:
            risk_tier = "MODERATE RISK"
        else:
            risk_tier = "HIGH RISK VULNERABLE"
            
        result = {
            "status": "success",
            "prediction": "Vulnerable" if pred_label == 1 else "Safe (Non-Vulnerable)",
            "prediction_label": pred_label,
            "risk_tier": risk_tier,
            "vulnerability_probability": round(vuln_prob * 100, 2),
            "safe_probability": round(safe_prob * 100, 2),
            "graph_stats": {
                "nodes_count": len(nodes),
                "edges_count": len(edges),
                "node_types_breakdown": {
                    nt: sum(1 for n in nodes if n["label"] == nt)
                    for nt in set(n["label"] for n in nodes)
                },
                "edge_types_breakdown": {
                    et: sum(1 for _, _, etype in edges if etype == et)
                    for et in set(etype for _, _, etype in edges)
                }
            }
        }
        
        if verbose:
            print("\n" + "=" * 65)
            print("🛡️  DEVIGN VULNERABILITY DETECTION RESULT")
            print("=" * 65)
            print(f"Prediction:            {result['prediction'].upper()}")
            print(f"Risk Assessment:       {risk_tier}")
            print(f"Vulnerability Risk:    {result['vulnerability_probability']}%")
            print(f"Confidence (Safe):     {result['safe_probability']}%")
            print(f"CPG Complexity:        {result['graph_stats']['nodes_count']} nodes, {result['graph_stats']['edges_count']} edges")
            print("=" * 65)
            
        return result
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Predict vulnerability in C/C++ source code using trained Graph Transformer.")
    parser.add_argument("--file", type=str, help="Path to a C/C++ source file")
    parser.add_argument("--code", type=str, help="Raw C/C++ source code string")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_graph_transformer.pt")
    
    args = parser.parse_args()
    
    if not args.file and not args.code:
        code = """
int process_packet(char *user_input, int length) {
    char buffer[64];
    if (length > 0) {
        // Unsafe memory copy without bounds check
        strcpy(buffer, user_input);
        return 0;
    }
    return -1;
}
"""
        print("[INFO] No input provided. Using sample C function (Buffer Overflow pattern):")
        print(code)
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            code = f.read()
    else:
        code = args.code
        
    model, vocab, device, threshold = load_trained_model(args.checkpoint)
    predict_c_code(code, model, vocab, device, threshold=threshold, verbose=True)

if __name__ == "__main__":
    main()
