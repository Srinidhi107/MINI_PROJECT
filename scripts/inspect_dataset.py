import os
import sys
import pandas as pd
import numpy as np

# Force UTF-8 stdout if needed
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def inspect_split(df: pd.DataFrame, split_name: str):
    print("=" * 70)
    print(f"EXPLORATION & INSPECTION: {split_name.upper()} SPLIT")
    print("=" * 70)
    print(f"Total samples: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    # 1. Missing Values
    print("\n[1] Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("  No missing values found!")
    
    # 2. Class Distribution (target)
    print("\n[2] Target Class Distribution:")
    # Normalize target to boolean/int representation
    target_counts = df['target'].value_counts()
    target_pct = df['target'].value_counts(normalize=True) * 100
    for val in target_counts.index:
        label_str = "Vulnerable (1/True)" if (val == 1 or val is True or str(val).lower() == 'true') else "Safe (0/False)"
        print(f"  {label_str}: {target_counts[val]:,} ({target_pct[val]:.2f}%)")
        
    # 3. Project Distribution
    if 'project' in df.columns:
        print("\n[3] Project Distribution:")
        proj_counts = df['project'].value_counts()
        for proj, cnt in proj_counts.items():
            print(f"  - {proj}: {cnt:,} ({cnt/len(df)*100:.2f}%)")
            
    # 4. Source Code Statistics (Length, Lines, Tokens)
    print("\n[4] Code Length Metrics:")
    df['char_len'] = df['func'].astype(str).str.len()
    df['line_count'] = df['func'].astype(str).apply(lambda x: len(x.splitlines()))
    df['token_approx'] = df['func'].astype(str).apply(lambda x: len(x.split()))
    
    stats_df = df[['char_len', 'line_count', 'token_approx']].describe().round(1)
    print(stats_df)
    
    # 5. Duplicate Functions
    duplicates = df.duplicated(subset=['func']).sum()
    print(f"\n[5] Duplicate Functions in {split_name}: {duplicates:,} duplicates found")

def main():
    train_path = "data/processed/train.csv"
    val_path = "data/processed/validation.csv"
    test_path = "data/processed/test.csv"
    
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found. Please run download_dataset.py first.")
        return
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    inspect_split(train_df, "train")
    inspect_split(val_df, "validation")
    inspect_split(test_df, "test")
    
    # Cross-split duplicates
    all_funcs = pd.concat([train_df[['func', 'target']], val_df[['func', 'target']], test_df[['func', 'target']]])
    cross_dupes = all_funcs.duplicated(subset=['func']).sum()
    print("\n" + "=" * 70)
    print(f"Cross-Split Duplicates across all 27,318 functions: {cross_dupes:,}")
    print("=" * 70)

if __name__ == "__main__":
    main()
