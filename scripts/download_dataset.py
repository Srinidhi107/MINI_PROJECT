import os
# pyrefly: ignore [missing-import]
from datasets import load_dataset
import pandas as pd

def main():
    print("=" * 60)
    print("Downloading Devign dataset from HuggingFace (DetectVul/devign)...")
    print("=" * 60)
    
    # Load dataset
    dataset = load_dataset("DetectVul/devign")
    
    print("\nDataset structure:")
    print(dataset)
    
    print("\n" + "=" * 60)
    print("Splits Summary:")
    print("=" * 60)
    for split_name in dataset.keys():
        print(f"Split [{split_name}]: {len(dataset[split_name])} samples")
        print(f"Columns: {dataset[split_name].column_names}")
    
    print("\n" + "=" * 60)
    print("Inspecting first sample from 'train' split:")
    print("=" * 60)
    first_sample = dataset["train"][0]
    for key, value in first_sample.items():
        if key == "func":
            # Show preview of function
            preview = value[:200] + "..." if len(value) > 200 else value
            print(f"  {key} (preview):\n{preview}\n")
        else:
            print(f"  {key}: {value}")
            
    # Also save to data/raw/devign or data/processed if needed
    os.makedirs("data/raw/devign", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    
    print("\n" + "=" * 60)
    print("Saving processed CSVs to data/processed/ ...")
    print("=" * 60)
    for split_name in dataset.keys():
        df = dataset[split_name].to_pandas()
        csv_path = f"data/processed/{split_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved {split_name} ({len(df)} rows) -> {csv_path}")

    print("\nDataset download and initial inspection completed successfully!")

if __name__ == "__main__":
    main()
