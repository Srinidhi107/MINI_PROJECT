import os
import sys
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import DevignGraphDataset  
from src.graph_transformer import GraphTransformer
from src.utils import load_config, compute_metrics

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch.utils.data import DataLoader

def collate_fn(batch):
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)

def plot_confusion_matrix(cm: list, output_path: str):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Safe (0)', 'Vulnerable (1)'],
                yticklabels=['Safe (0)', 'Vulnerable (1)'])
    plt.title('Confusion Matrix - Vulnerability Detection')
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to '{output_path}'", flush=True)

def plot_roc_curve(y_true: np.ndarray, y_probs: np.ndarray, roc_auc: float, output_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved ROC curve plot to '{output_path}'", flush=True)

def plot_pr_curve(y_true: np.ndarray, y_probs: np.ndarray, pr_auc: float, output_path: str):
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_curve, precision_curve, color='teal', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    no_skill = float((y_true == 1).mean())
    plt.plot([0, 1], [no_skill, no_skill], color='gray', lw=1.5, linestyle='--', label=f'Baseline ({no_skill:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (PR)')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved PR curve plot to '{output_path}'", flush=True)

def plot_metrics_barchart(metrics: dict, output_path: str):
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC']
    values = [
        metrics['accuracy'] * 100,
        metrics['precision'] * 100,
        metrics['recall'] * 100,
        metrics['f1'] * 100,
        metrics['roc_auc'] * 100,
        metrics['pr_auc'] * 100
    ]
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728', '#9467bd', '#8c564b']
    
    plt.figure(figsize=(8.5, 5))
    bars = plt.bar(metric_names, values, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    plt.ylim(0, 100)
    plt.ylabel('Score (%)', fontsize=12)
    plt.title('Graph Transformer Test Performance Metrics Overview', fontsize=13, fontweight='bold', pad=15)
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5, f"{val:.1f}%",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
                 
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved metrics summary bar chart to '{output_path}'", flush=True)

def plot_probability_distribution(y_true: np.ndarray, y_probs: np.ndarray, threshold: float, output_path: str):
    plt.figure(figsize=(7, 5))
    sns.histplot(y_probs[y_true == 0], color='blue', label='Safe (0)', kde=True, stat="density", bins=25, alpha=0.35)
    sns.histplot(y_probs[y_true == 1], color='red', label='Vulnerable (1)', kde=True, stat="density", bins=25, alpha=0.35)
    plt.axvline(x=threshold, color='black', linestyle='--', lw=2, label=f'Threshold ({threshold:.2f})')
    plt.title('Predicted Probability Distribution (Safe vs. Vulnerable)', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Vulnerability Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved probability distribution plot to '{output_path}'", flush=True)

def evaluate_model(checkpoint_path: str, split: str = "test"):
    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device") == "cuda" else "cpu")
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' does not exist.", flush=True)
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_path = config["data"]["vocab_path"]
    max_seq_len = config["data"].get("max_seq_len", 128)
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
        
    # Dataset & Loader
    dataset = DevignGraphDataset(config["data"]["graph_dir"], split=split, max_seq_len=max_seq_len, preload=True)
    if len(dataset) == 0:
        print(f"No graphs found for split '{split}' in {config['data']['graph_dir']}", flush=True)
        return
        
    loader = DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    # Initialize Model
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
        pooling=m_cfg["pooling"],
        max_seq_len=max_seq_len
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    threshold = float(checkpoint.get("optimal_threshold", 0.50))
    print(f"Loaded checkpoint from '{checkpoint_path}' (Epoch {checkpoint.get('epoch', '?')}, Decision Threshold: {threshold:.2f})", flush=True)
    
    all_preds_prob = []
    all_targets = []
    all_func_ids = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            seq_tokens = getattr(batch, 'seq_tokens', None)
            code_vec = getattr(batch, 'code_vec', None)
            graph_vec = getattr(batch, 'graph_vec', None)
            codebert_vec = getattr(batch, 'codebert_vec', None)
            
            logits = model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                seq_tokens=seq_tokens,
                code_vec=code_vec,
                graph_vec=graph_vec,
                codebert_vec=codebert_vec
            )
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            targets = batch.y.view(-1).cpu().numpy()
            
            all_preds_prob.extend(probs)
            all_targets.extend(targets)
            if hasattr(batch, 'func_id'):
                all_func_ids.extend(batch.func_id)
                
    y_true = np.array(all_targets)
    y_probs = np.array(all_preds_prob)
    y_pred = (y_probs >= threshold).astype(int)
    
    metrics = compute_metrics(y_true, y_probs)
    # Compute metrics at the optimal threshold
    metrics_at_thresh = compute_metrics(y_true, y_pred)
    metrics["accuracy"] = metrics_at_thresh["accuracy"]
    metrics["precision"] = metrics_at_thresh["precision"]
    metrics["recall"] = metrics_at_thresh["recall"]
    metrics["f1"] = metrics_at_thresh["f1"]
    metrics["macro_f1"] = metrics_at_thresh["macro_f1"]
    metrics["confusion_matrix"] = metrics_at_thresh["confusion_matrix"]
    metrics["decision_threshold"] = threshold
    
    print("\n" + "=" * 70, flush=True)
    print(f"FINAL EVALUATION ON '{split.upper()}' SPLIT (Threshold = {threshold:.2f})", flush=True)
    print("=" * 70, flush=True)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"Macro F1:  {metrics['macro_f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:    {metrics['pr_auc']:.4f}")
    print("\nDetailed Classification Report:", flush=True)
    print(classification_report(y_true, y_pred, target_names=['Safe (0)', 'Vulnerable (1)']), flush=True)
    
    # Save Predictions CSV
    res_dir = config["output"]["results_dir"]
    pred_dir = os.path.join(res_dir, "predictions")
    plots_dir = os.path.join(res_dir, "plots")
    metrics_dir = os.path.join(res_dir, "metrics")
    
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    preds_df = pd.DataFrame({
        "function_id": all_func_ids if len(all_func_ids) == len(y_true) else [f"{split}_{i}" for i in range(len(y_true))],
        "true_label": y_true,
        "predicted_label": y_pred,
        "vuln_probability": y_probs
    })
    pred_csv_path = os.path.join(pred_dir, f"{split}_predictions.csv")
    preds_df.to_csv(pred_csv_path, index=False)
    print(f"Saved prediction results to '{pred_csv_path}'", flush=True)
    
    # Save Metrics JSON
    metrics_file = os.path.join(metrics_dir, f"{split}_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics summary to '{metrics_file}'", flush=True)
    
    # Generate Plots
    plot_confusion_matrix(metrics['confusion_matrix'], os.path.join(plots_dir, f"{split}_confusion_matrix.png"))
    plot_roc_curve(y_true, y_probs, metrics['roc_auc'], os.path.join(plots_dir, f"{split}_roc_curve.png"))
    plot_pr_curve(y_true, y_probs, metrics['pr_auc'], os.path.join(plots_dir, f"{split}_pr_curve.png"))
    plot_metrics_barchart(metrics, os.path.join(plots_dir, f"{split}_metrics_barchart.png"))
    plot_probability_distribution(y_true, y_probs, threshold, os.path.join(plots_dir, f"{split}_probability_distribution.png"))

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Graph Transformer on vulnerability detection.")
    parser.add_argument("--checkpoint", type=str, default="models/checkpoints/best_graph_transformer.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint, split=args.split)

if __name__ == "__main__":
    main()
