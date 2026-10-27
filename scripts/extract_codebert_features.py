import os
import sys
import re
import glob
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    print("=" * 70, flush=True)
    print("EXTRACTING PRE-TRAINED CODEBERT (768-DIM) EMBEDDINGS", flush=True)
    print("=" * 70, flush=True)

    # 1. Load CSV data for code lookup
    print("Loading CSV metadata for function code mapping...", flush=True)
    df_train = pd.read_csv("data/processed/train.csv")
    df_val = pd.read_csv("data/processed/validation.csv")
    df_test = pd.read_csv("data/processed/test.csv")
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    id_to_row = {int(r["id"]): r for _, r in df_all.iterrows()}
    print(f"Loaded {len(id_to_row):,} functions into code dictionary.", flush=True)

    # 2. Load CodeBERT Tokenizer & Model
    print("Loading pre-trained Microsoft CodeBERT...", flush=True)
    model_name = "microsoft/codebert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    num_threads = min(8, os.cpu_count() or 4)
    torch.set_num_threads(num_threads)
    print(f"CodeBERT ready on CPU using {num_threads} threads.", flush=True)

    # 3. Process each split
    output_path = "data/graphs/codebert_embeddings.pt"
    embeddings_dict = {}
    if os.path.exists(output_path):
        try:
            embeddings_dict = torch.load(output_path, weights_only=False)
            print(f"Loaded {len(embeddings_dict)} existing embeddings from cache.", flush=True)
        except Exception:
            embeddings_dict = {}

    batch_size = 32
    max_len = 128

    splits = ["train", "val", "test"]
    for split in splits:
        split_dir = os.path.join("data/graphs", split)
        pt_files = sorted(glob.glob(os.path.join(split_dir, "*.pt")))
        print(f"\nProcessing {split.upper()} ({len(pt_files)} graphs)...", flush=True)

        # Filter out already computed
        pending = []
        for f in pt_files:
            fname = os.path.basename(f)
            func_id = os.path.splitext(fname)[0]
            if func_id not in embeddings_dict:
                m = re.search(r"_(\d+)", fname)
                if m:
                    idx_num = int(m.group(1))
                    if idx_num in id_to_row:
                        row = id_to_row[idx_num]
                        code = str(row.get("func_clean", row.get("func", "")))
                        pending.append((func_id, code))
                    else:
                        pending.append((func_id, ""))
                else:
                    pending.append((func_id, ""))

        print(f"  Pending embeddings to compute: {len(pending)}", flush=True)

        for i in range(0, len(pending), batch_size):
            chunk = pending[i:i + batch_size]
            func_ids = [c[0] for c in chunk]
            codes = [c[1] if c[1] and c[1] != "nan" else "void empty() {}" for c in chunk]

            inputs = tokenizer(
                codes,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(**inputs)
                # CLS token embedding: [batch, 768]
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()

            for f_id, emb in zip(func_ids, cls_embeddings):
                embeddings_dict[f_id] = emb.unsqueeze(0) # [1, 768]

            if (i // batch_size) % 20 == 0 or i + batch_size >= len(pending):
                print(f"  Processed {min(i + batch_size, len(pending))}/{len(pending)} samples...", flush=True)

    # 4. Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(embeddings_dict, output_path)
    print("\n" + "=" * 70, flush=True)
    print(f"Saved total {len(embeddings_dict)} CodeBERT embeddings to '{output_path}'", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
