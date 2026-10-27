import os
import sys
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def clean_source_code(code: str) -> str:
    """Normalize line endings and strip trailing whitespace."""
    if not isinstance(code, str):
        return ""
    # Standardize line endings to \n
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    # Clean trailing whitespaces per line
    lines = [line.rstrip() for line in code.split("\n")]
    return "\n".join(lines).strip()

def preprocess_split(input_path: str, output_path: str, split_name: str) -> pd.DataFrame:
    print(f"\nProcessing {split_name} split from '{input_path}'...")
    df = pd.read_csv(input_path)
    
    initial_count = len(df)
    
    # 1. Standardize Target Labels (0 = Non-vulnerable, 1 = Vulnerable)
    df['target'] = df['target'].apply(lambda x: 1 if (x is True or x == 1 or str(x).lower() == 'true') else 0)
    
    # 2. Clean Source Code
    df['func'] = df['func'].apply(clean_source_code)
    
    # Filter out any empty functions
    df = df[df['func'].str.len() > 0].copy()
    
    # 3. Add explicit function identifier if not present or ensure unique ID
    df['function_id'] = [f"{split_name}_{i}" for i in range(len(df))]
    
    # 4. Extract length metrics
    df['char_count'] = df['func'].str.len()
    df['line_count'] = df['func'].apply(lambda x: len(x.splitlines()))
    df['token_count'] = df['func'].apply(lambda x: len(x.split()))
    
    # Reorder key columns
    cols_order = ['function_id', 'id', 'target', 'project', 'commit_id', 'line_count', 'token_count', 'char_count', 'func']
    # Include other auxiliary columns if present
    remaining_cols = [c for c in df.columns if c not in cols_order]
    df = df[cols_order + remaining_cols]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"  -> Kept {len(df):,}/{initial_count:,} functions.")
    print(f"  -> Target balance: Safe(0)={sum(df['target']==0):,}, Vuln(1)={sum(df['target']==1):,}")
    print(f"  -> Saved clean dataset to '{output_path}'")
    
    return df

def main():
    print("=" * 70)
    print("STEP 2: PREPROCESSING DEVIGN DATASET")
    print("=" * 70)
    
    raw_processed_dir = "data/processed"
    
    train_df = preprocess_split(
        f"{raw_processed_dir}/train.csv", 
        f"{raw_processed_dir}/train_clean.csv", 
        "train"
    )
    val_df = preprocess_split(
        f"{raw_processed_dir}/validation.csv", 
        f"{raw_processed_dir}/val_clean.csv", 
        "val"
    )
    test_df = preprocess_split(
        f"{raw_processed_dir}/test.csv", 
        f"{raw_processed_dir}/test_clean.csv", 
        "test"
    )
    
    print("\n" + "=" * 70)
    print("Preprocessing completed successfully!")
    print(f"Total processed samples: {len(train_df) + len(val_df) + len(test_df):,}")
    print("=" * 70)

if __name__ == "__main__":
    main()
