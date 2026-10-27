import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def extract_source_files(csv_path: str, output_dir: str, split_name: str, limit: int = None):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' does not exist.")
        return
        
    df = pd.read_csv(csv_path)
    total_in_df = len(df)
    
    if limit is not None and limit < total_in_df:
        # Sample evenly across classes (stratified sample) if possible
        vulnerable = df[df['target'] == 1].head(limit // 2)
        safe = df[df['target'] == 0].head(limit - len(vulnerable))
        df_subset = pd.concat([vulnerable, safe]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Extracting sample subset of {len(df_subset)} functions for '{split_name}' split (from {total_in_df} total)...")
    else:
        df_subset = df
        print(f"Extracting all {len(df_subset)} functions for '{split_name}' split...")

    os.makedirs(output_dir, exist_ok=True)
    
    manifest = []
    
    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc=f"Writing {split_name} .c files"):
        func_id = str(row.get('function_id', f"{split_name}_{idx}"))
        source_code = str(row['func'])
        
        # File path
        filename = f"{func_id}.c"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(source_code)
            
        manifest.append({
            'function_id': func_id,
            'filename': filename,
            'filepath': filepath,
            'target': int(row['target']),
            'project': row.get('project', ''),
            'commit_id': row.get('commit_id', ''),
            'line_count': row.get('line_count', len(source_code.splitlines())),
            'char_count': row.get('char_count', len(source_code))
        })
        
    # Save manifest mapping for quick lookup
    manifest_df = pd.DataFrame(manifest)
    manifest_path = os.path.join(output_dir, f"manifest_{split_name}.csv")
    manifest_df.to_csv(manifest_path, index=False)
    
    print(f"Completed '{split_name}': {len(manifest_df)} files written to '{output_dir}'.")
    print(f"Manifest saved to: '{manifest_path}'")

def main():
    parser = argparse.ArgumentParser(description="Extract C functions from Devign CSVs into individual .c files.")
    parser.add_argument("--limit", type=int, default=50, help="Number of functions to extract per split (default: 50). Use --all to extract all.")
    parser.add_argument("--all", action="store_true", help="Extract all functions in the dataset.")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"], help="Which split to extract.")
    
    args = parser.parse_args()
    
    limit = None if args.all else args.limit
    
    print("=" * 70)
    print("STEP 3: SOURCE CODE EXTRACTION (PHASED)")
    print(f"Extraction mode: {'ALL samples' if args.all else f'Sample subset (limit={limit} per split)'}")
    print("=" * 70)
    
    base_data_dir = "data/raw/devign"
    processed_dir = "data/processed"
    
    splits_to_process = ["train", "val", "test"] if args.split == "all" else [args.split]
    
    for split in splits_to_process:
        csv_file = os.path.join(processed_dir, f"{split}_clean.csv")
        target_dir = os.path.join(base_data_dir, split)
        extract_source_files(csv_file, target_dir, split, limit=limit)
        
    print("\n" + "=" * 70)
    print("Source code extraction phase completed successfully!")
    print("=" * 70)

if __name__ == "__main__":
    main()
