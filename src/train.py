import os
import sys
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import DevignGraphDataset
from src.graph_transformer import GraphTransformer
from src.utils import set_seed, load_config, compute_metrics

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch.utils.data import DataLoader

from src.loss import FocalLossWithSmoothing

def collate_fn(batch):
    from torch_geometric.data import Batch
    return Batch.from_data_list(batch)

def train_epoch(model, batches, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    total_graphs = 0
    
    for batch in batches:
        optimizer.zero_grad()
        
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
        loss = criterion(logits, batch.y.view(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        n_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.y.size(0)
        total_loss += loss.item() * n_graphs
        total_graphs += n_graphs
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        all_preds.extend(probs)
        all_targets.extend(batch.y.view(-1).cpu().numpy())
        
    avg_loss = total_loss / max(total_graphs, 1)
    metrics = compute_metrics(np.array(all_targets), np.array(all_preds))
    metrics["loss"] = avg_loss
    return metrics

def eval_epoch(model, batches, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    total_graphs = 0
    
    with torch.no_grad():
        for batch in batches:
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
            loss = criterion(logits, batch.y.view(-1))
            
            n_graphs = batch.num_graphs if hasattr(batch, 'num_graphs') else batch.y.size(0)
            total_loss += loss.item() * n_graphs
            total_graphs += n_graphs
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(batch.y.view(-1).cpu().numpy())
            
    avg_loss = total_loss / max(total_graphs, 1)
    y_true = np.array(all_targets)
    y_probs = np.array(all_preds)
    
    # Threshold search on validation split to optimize Balanced Macro F1
    best_thresh = 0.50
    best_f1 = 0.0
    for t in np.linspace(0.35, 0.65, 31):
        preds = (y_probs >= t).astype(int)
        f = compute_metrics(y_true, preds)["macro_f1"]
        if f > best_f1:
            best_f1 = f
            best_thresh = float(t)
            
    metrics = compute_metrics(y_true, y_probs)
    preds_optimal = (y_probs >= best_thresh).astype(int)
    opt_metrics = compute_metrics(y_true, preds_optimal)
    
    metrics["loss"] = avg_loss
    metrics["optimal_threshold"] = best_thresh
    metrics["optimal_accuracy"] = opt_metrics["accuracy"]
    metrics["optimal_macro_f1"] = opt_metrics["macro_f1"]
    return metrics

def plot_training_history(history: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Loss Plot
    plt.figure(figsize=(7, 5))
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss', color='royalblue', lw=2)
    if 'val_loss' in history.columns:
        plt.plot(history['epoch'], history['val_loss'], label='Val Loss', color='crimson', lw=2)
    plt.title('Training & Validation Loss Curve', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    loss_plot_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    
    # 2. Accuracy & F1 Plot
    plt.figure(figsize=(7, 5))
    plt.plot(history['epoch'], history['train_accuracy'], label='Train Accuracy', color='royalblue', lw=2)
    plt.plot(history['epoch'], history['train_f1'], label='Train F1', color='cornflowerblue', linestyle='--', lw=2)
    if 'val_accuracy' in history.columns:
        plt.plot(history['epoch'], history['val_accuracy'], label='Val Accuracy', color='crimson', lw=2)
        plt.plot(history['epoch'], history['val_f1'], label='Val F1', color='salmon', linestyle='--', lw=2)
    plt.title('Accuracy and F1-Score Progression', fontsize=12)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    metric_plot_path = os.path.join(output_dir, "training_metrics.png")
    plt.savefig(metric_plot_path, dpi=300)
    plt.close()

def main():
    config = load_config()
    set_seed(config.get("seed", 42))
    
    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    
    device_name = config.get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    print(f"Using compute device: {device} ({num_threads} CPU threads)", flush=True)
    
    graph_dir = config["data"]["graph_dir"]
    vocab_path = config["data"]["vocab_path"]
    max_seq_len = config["data"].get("max_seq_len", 128)
    
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary file '{vocab_path}' not found. Please run build_graphs.py first.", flush=True)
        return
        
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    
    # Load Datasets
    print("Preloading graphs into memory for high-speed training...", flush=True)
    train_dataset = DevignGraphDataset(graph_dir, split="train", max_seq_len=max_seq_len, preload=True)
    val_dataset = DevignGraphDataset(graph_dir, split="val", max_seq_len=max_seq_len, preload=True)
    
    print(f"Loaded Graphs -> Train: {len(train_dataset)} graphs, Validation: {len(val_dataset)} graphs", flush=True)
    if len(train_dataset) == 0:
        print("Error: Train dataset is empty. Please run build_graphs.py first.", flush=True)
        return
        
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn) if len(val_dataset) > 0 else None
    
    # Pre-collate batches in RAM for ultra-fast epoch iterations
    print("Pre-collating batches into memory for ultra-fast epoch iterations...", flush=True)
    train_batches = [batch.to(device) for batch in train_loader]
    val_batches = [batch.to(device) for batch in val_loader] if val_loader else []
    print(f"Ready: {len(train_batches)} Train Batches and {len(val_batches)} Validation Batches pre-collated.", flush=True)
    
    # Initialize Multi-Modal Graph Transformer Model
    m_cfg = config["model"]
    model = GraphTransformer(
        vocab_size=vocab_size,
        node_type_dim=config["data"].get("node_type_dim", 17),
        max_tokens_per_node=config["data"].get("max_tokens_per_node", 10),
        num_sec_flags=6,
        hidden_dim=m_cfg["hidden_dim"],
        num_layers=m_cfg["num_layers"],
        num_heads=m_cfg["num_heads"],
        num_edge_types=config["data"]["edge_types"],
        num_classes=config["data"]["num_classes"],
        dropout=m_cfg["dropout"],
        drop_edge_rate=m_cfg.get("drop_edge", 0.15),
        pooling=m_cfg["pooling"],
        max_seq_len=max_seq_len
    ).to(device)
    
    print(f"Initialized Dual-Stream Graph Transformer ({sum(p.numel() for p in model.parameters()):,} parameters).", flush=True)
    
    # Loss & Optimizer
    label_smoothing = config.get("training", {}).get("label_smoothing", 0.05)
    criterion = FocalLossWithSmoothing(alpha=0.52, gamma=1.5, label_smoothing=label_smoothing)
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    epochs = config["training"]["epochs"]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5
    )
    
    # Checkpoint setup
    checkpoint_dir = config["output"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, config["output"]["best_model_name"])
    
    results_dir = config["output"]["results_dir"]
    metrics_dir = os.path.join(results_dir, "metrics")
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    best_val_score = -1.0
    history_records = []
    
    print("\n" + "=" * 70, flush=True)
    print("STARTING GRAPH TRANSFORMER TRAINING LOOP", flush=True)
    print(f"Readout: {m_cfg['pooling']} | DropEdge: {m_cfg.get('drop_edge', 0.15)} | Label Smoothing: {label_smoothing}", flush=True)
    print("=" * 70, flush=True)
    
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_batches, optimizer, criterion, device)
        scheduler.step()
        
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "lr": optimizer.param_groups[0]["lr"]
        }
        
        if val_batches and len(val_batches) > 0:
            val_metrics = eval_epoch(model, val_batches, criterion, device)
            val_score = (val_metrics["accuracy"] + val_metrics["macro_f1"] + val_metrics["roc_auc"]) / 3.0
            
            record.update({
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_roc_auc": val_metrics["roc_auc"],
                "optimal_threshold": val_metrics["optimal_threshold"]
            })
            
            print(f"Epoch {epoch:02d}/{epochs:02d} | "
                  f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.3f}, F1: {train_metrics['f1']:.3f} | "
                  f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.3f}, Macro-F1: {val_metrics['macro_f1']:.3f}, AUC: {val_metrics['roc_auc']:.3f} | "
                  f"OptThresh: {val_metrics['optimal_threshold']:.2f} (OptAcc: {val_metrics['optimal_accuracy']:.3f}) | LR: {optimizer.param_groups[0]['lr']:.6f}", flush=True)
                  
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "optimal_threshold": val_metrics.get("optimal_threshold", 0.5),
                    "config": config
                }, best_checkpoint_path)
                print(f"  --> Saved new best checkpoint to '{best_checkpoint_path}' (Val Balanced Score: {val_score:.4f})", flush=True)
        else:
            print(f"Epoch {epoch:02d}/{epochs:02d} | "
                  f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.3f}, F1: {train_metrics['f1']:.3f}", flush=True)
                  
        history_records.append(record)
        
    history_df = pd.DataFrame(history_records)
    history_csv_path = os.path.join(metrics_dir, "training_history.csv")
    history_df.to_csv(history_csv_path, index=False)
    print(f"\nSaved training history to '{history_csv_path}'", flush=True)
    
    plot_training_history(history_df, plots_dir)

    print("\n" + "=" * 70, flush=True)
    print(f"Training completed successfully! Best Validation Score: {best_val_score:.4f}", flush=True)
    print(f"Best model saved at: {best_checkpoint_path}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
