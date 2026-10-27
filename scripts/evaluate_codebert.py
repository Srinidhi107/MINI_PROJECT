import os
import sys
import re
import glob
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils import compute_metrics, set_seed

class CodeTextDataset(Dataset):
    def __init__(self, texts, labels, func_ids):
        self.texts = texts
        self.labels = labels
        self.func_ids = func_ids

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

def evaluate():
    checkpoint_path = "models/checkpoints/best_codebert.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint '{checkpoint_path}' not found yet.")
        sys.exit(1)

    print("=" * 70, flush=True)
    print("EVALUATING CODEBERT ON DEVIGN TEST SET (1,500 SAMPLES)", flush=True)
    print("=" * 70, flush=True)

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    print(f"Using device: {device} ({num_threads} threads)", flush=True)

    # 1. Load test metadata
    df_train = pd.read_csv("data/processed/train.csv")
    df_val = pd.read_csv("data/processed/validation.csv")
    df_test = pd.read_csv("data/processed/test.csv")
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    id_to_row = {int(r["id"]): r for _, r in df_all.iterrows()}

    test_texts, test_labels, test_ids = load_data_split("test", id_to_row)
    print(f"Loaded {len(test_texts)} test functions.", flush=True)

    # 2. Tokenizer & Model
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", 1)
    threshold = checkpoint.get("optimal_threshold", 0.50)
    print(f"Loaded checkpoint from Epoch {epoch} with optimal threshold {threshold:.2f}", flush=True)

    # 3. Test DataLoader
    test_ds = CodeTextDataset(test_texts, test_labels, test_ids)
    test_loader = DataLoader(
        test_ds,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer, max_length=128)
    )

    # 4. Inference
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

    y_true = np.array(test_targets)
    y_probs = np.array(test_probs)
    y_pred = (y_probs >= threshold).astype(int)

    test_metrics = compute_metrics(y_true, y_probs)
    thresh_metrics = compute_metrics(y_true, y_pred)

    final_results = {
        "model": "Fine-Tuned Microsoft CodeBERT",
        "epoch_evaluated": epoch,
        "accuracy": thresh_metrics["accuracy"],
        "precision": thresh_metrics["precision"],
        "recall": thresh_metrics["recall"],
        "f1": thresh_metrics["f1"],
        "macro_f1": thresh_metrics["macro_f1"],
        "roc_auc": test_metrics["roc_auc"],
        "pr_auc": test_metrics["pr_auc"],
        "confusion_matrix": thresh_metrics["confusion_matrix"],
        "decision_threshold": threshold
    }

    print("\n" + "=" * 70)
    print(f"FINAL TEST ACCURACY:  {final_results['accuracy']:.4f} ({final_results['accuracy']*100:.2f}%)")
    print(f"FINAL TEST PRECISION: {final_results['precision']:.4f}")
    print(f"FINAL TEST RECALL:    {final_results['recall']:.4f}")
    print(f"FINAL TEST F1 SCORE:  {final_results['f1']:.4f}")
    print(f"FINAL TEST MACRO F1:  {final_results['macro_f1']:.4f}")
    print(f"FINAL TEST ROC-AUC:   {final_results['roc_auc']:.4f}")
    print(f"FINAL TEST PR-AUC:    {final_results['pr_auc']:.4f}")
    print("=" * 70)
    print("\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Safe (0)', 'Vulnerable (1)']))

    # Save outputs
    metrics_path = "results/metrics/codebert_test_metrics.json"
    preds_path = "results/predictions/codebert_test_predictions.csv"
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(preds_path), exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nSaved test metrics to '{metrics_path}'")

    preds_df = pd.DataFrame({
        "function_id": test_out_ids,
        "true_label": y_true,
        "predicted_label": y_pred,
        "vuln_probability": y_probs
    })
    preds_df.to_csv(preds_path, index=False)
    print(f"Saved test predictions to '{preds_path}'")

if __name__ == "__main__":
    evaluate()
