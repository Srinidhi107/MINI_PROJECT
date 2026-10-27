import os
import sys
import re
import glob
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import compute_metrics, set_seed

class CodeTextDataset(Dataset):
    def __init__(self, texts, labels, func_ids, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.func_ids = func_ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if not text or text == "nan":
            text = "void empty() {}"
        return {
            "text": text,
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "func_id": self.func_ids[idx]
        }

def collate_fn(batch, tokenizer, max_length=128):
    texts = [item["text"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    func_ids = [item["func_id"] for item in batch]
    
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc["labels"] = labels
    enc["func_ids"] = func_ids
    return enc

def load_data_split(split, id_to_row):
    files = sorted(glob.glob(os.path.join("data/graphs", split, "*.pt")))
    texts, labels, func_ids = [], [], []
    for f in files:
        fname = os.path.basename(f)
        f_id = os.path.splitext(fname)[0]
        d = torch.load(f, weights_only=False)
        m = re.search(r"_(\d+)", fname)
        code = ""
        if m:
            idx_num = int(m.group(1))
            if idx_num in id_to_row:
                row = id_to_row[idx_num]
                code = str(row.get("func_clean", row.get("func", "")))
        y = int(d.y.item()) if hasattr(d, "y") else 0
        texts.append(code)
        labels.append(y)
        func_ids.append(f_id)
    return texts, labels, func_ids

def main():
    print("=" * 70, flush=True)
    print("DIRECT FINE-TUNING OF MICROSOFT CODEBERT FOR DEVIGN VULNERABILITY DETECTION", flush=True)
    print("=" * 70, flush=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    print(f"Using device: {device} ({num_threads} CPU threads)", flush=True)

    # 1. Load CSV metadata
    print("Loading source code metadata...", flush=True)
    df_train = pd.read_csv("data/processed/train.csv")
    df_val = pd.read_csv("data/processed/validation.csv")
    df_test = pd.read_csv("data/processed/test.csv")
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    id_to_row = {int(r["id"]): r for _, r in df_all.iterrows()}

    # 2. Load splits
    print("Mapping dataset splits to C function code...", flush=True)
    train_texts, train_labels, train_ids = load_data_split("train", id_to_row)
    val_texts, val_labels, val_ids = load_data_split("val", id_to_row)
    test_texts, test_labels, test_ids = load_data_split("test", id_to_row)

    print(f"Loaded: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}", flush=True)

    # 3. Model & Tokenizer
    model_name = "microsoft/codebert-base"
    print(f"Loading tokenizer & sequence classification model: {model_name}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # Freeze bottom 11 layers for ultra-fast CPU adaptation & anti-forgetting
    print("Configuring layers for fine-tuning...", flush=True)
    for param in model.roberta.embeddings.parameters():
        param.requires_grad = False
    for i in range(11):
        for param in model.roberta.encoder.layer[i].parameters():
            param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Fine-tuning Layer 11 + Classifier: {trainable_params:,} / {total_params:,} parameters active.", flush=True)

    # 4. DataLoaders
    batch_size = 32
    max_length = 96
    train_ds = CodeTextDataset(train_texts, train_labels, train_ids, tokenizer, max_length)
    val_ds = CodeTextDataset(val_texts, val_labels, val_ids, tokenizer, max_length)
    test_ds = CodeTextDataset(test_texts, test_labels, test_ids, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, tokenizer, max_length))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, max_length))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, tokenizer, max_length))

    # 5. Optimizer & Scheduler
    epochs = 1
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad], "lr": 1e-4},
        {"params": [p for n, p in model.named_parameters() if "classifier" not in n and p.requires_grad], "lr": 2e-5}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Class balance weights
    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    pos_weight = float(num_neg) / float(max(num_pos, 1))
    weights = torch.tensor([1.0, pos_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    checkpoint_dir = "models/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_codebert.pt")

    # 6. Training Loop
    print("\n" + "=" * 70, flush=True)
    print("STARTING CODEBERT FINE-TUNING (5 EPOCHS)", flush=True)
    print("=" * 70, flush=True)

    best_val_score = -1.0
    best_threshold = 0.50

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(labels)
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            train_preds.extend(probs)
            train_targets.extend(labels.cpu().numpy())

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(train_loader):
                cur_loss = train_loss / ((batch_idx + 1) * batch_size)
                print(f"  [Epoch {epoch}] Batch {batch_idx + 1:03d}/{len(train_loader)} | Running Loss: {cur_loss:.4f}", flush=True)

        scheduler.step()
        epoch_train_loss = train_loss / len(train_texts)
        train_acc = float(((np.array(train_preds) >= 0.50).astype(int) == np.array(train_targets)).mean())

        # Validation
        model.eval()
        val_loss = 0.0
        val_probs, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)

                val_loss += loss.item() * len(labels)
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_targets.extend(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_texts)
        y_val_true = np.array(val_targets)
        y_val_probs = np.array(val_probs)

        # Threshold calibration
        best_t, best_f1 = 0.50, 0.0
        for t in np.linspace(0.35, 0.65, 31):
            preds = (y_val_probs >= t).astype(int)
            m = compute_metrics(y_val_true, preds)
            if m["macro_f1"] > best_f1:
                best_f1 = m["macro_f1"]
                best_t = float(t)

        val_metrics = compute_metrics(y_val_true, y_val_probs)
        opt_preds = (y_val_probs >= best_t).astype(int)
        opt_acc = float((opt_preds == y_val_true).mean())
        val_score = (opt_acc + best_f1 + val_metrics["roc_auc"]) / 3.0

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{epochs:02d} [{elapsed:.0f}s] | "
              f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {train_acc:.3f} | "
              f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {opt_acc:.3f}, Macro-F1: {best_f1:.3f}, AUC: {val_metrics['roc_auc']:.3f} | "
              f"OptThresh: {best_t:.2f}", flush=True)

        if val_score > best_val_score:
            best_val_score = val_score
            best_threshold = best_t
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimal_threshold": best_t,
                "val_score": val_score,
                "val_acc": opt_acc
            }, best_model_path)
            print(f"  --> Saved new best CodeBERT checkpoint to '{best_model_path}'", flush=True)

    # 7. Final Evaluation on Test Set
    print("\n" + "=" * 70, flush=True)
    print(f"FINAL EVALUATION ON TEST SET (1,500 SAMPLES) USING OPTIMAL THRESHOLD: {best_threshold:.2f}", flush=True)
    print("=" * 70, flush=True)

    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_probs, test_targets, test_out_ids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            test_probs.extend(probs)
            test_targets.extend(batch["labels"].cpu().numpy())
            test_out_ids.extend(batch["func_ids"])

    y_test_true = np.array(test_targets)
    y_test_probs = np.array(test_probs)
    y_test_pred = (y_test_probs >= best_threshold).astype(int)

    test_metrics = compute_metrics(y_test_true, y_test_probs)
    thresh_metrics = compute_metrics(y_test_true, y_test_pred)

    final_results = {
        "model": "Fine-Tuned Microsoft CodeBERT",
        "accuracy": thresh_metrics["accuracy"],
        "precision": thresh_metrics["precision"],
        "recall": thresh_metrics["recall"],
        "f1": thresh_metrics["f1"],
        "macro_f1": thresh_metrics["macro_f1"],
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"],
        "confusion_matrix": thresh_metrics["confusion_matrix"],
        "decision_threshold": best_threshold
    }

    print(f"Final Test Accuracy:  {final_results['accuracy']:.4f} ({final_results['accuracy']*100:.2f}%)")
    print(f"Final Test Precision: {final_results['precision']:.4f}")
    print(f"Final Test Recall:    {final_results['recall']:.4f}")
    print(f"Final Test F1 Score:  {final_results['f1']:.4f}")
    print(f"Final Test Macro F1:  {final_results['macro_f1']:.4f}")
    print(f"Final Test ROC-AUC:   {final_results['roc_auc']:.4f}")
    print(f"Final Test PR-AUC:    {final_results['pr_auc']:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test_true, y_test_pred, target_names=['Safe (0)', 'Vulnerable (1)']))

    # Save outputs
    res_dir = "results"
    metrics_path = os.path.join(res_dir, "metrics", "codebert_test_metrics.json")
    preds_path = os.path.join(res_dir, "predictions", "codebert_test_predictions.csv")
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(preds_path), exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved metrics to '{metrics_path}'")

    preds_df = pd.DataFrame({
        "function_id": test_out_ids,
        "true_label": y_test_true,
        "predicted_label": y_test_pred,
        "vuln_probability": y_test_probs
    })
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved predictions to '{preds_path}'")

    print("\n" + "=" * 70, flush=True)
    print("CODEBERT FINE-TUNING AND EVALUATION COMPLETED SUCCESSFULLY!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
