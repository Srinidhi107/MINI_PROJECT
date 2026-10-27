import os
import sys
import glob
import shutil
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
JOERN_DIR = os.path.join(PROJECT_ROOT, "tools", "joern", "joern-cli")
C2CPG_BAT = os.path.join(JOERN_DIR, "c2cpg.bat")
JOERN_EXPORT_BAT = os.path.join(JOERN_DIR, "joern-export.bat")

def generate_cpg_for_file(c_filepath: str, output_dot_path: str, temp_dir: str) -> bool:
    """Run c2cpg and joern-export on a single C file to generate a DOT CPG graph."""
    func_id = os.path.splitext(os.path.basename(c_filepath))[0]
    temp_bin = os.path.join(temp_dir, f"{func_id}.bin")
    temp_export_dir = os.path.join(temp_dir, f"export_{func_id}")
    
    # 1. Generate CPG binary using c2cpg.bat
    c2cpg_cmd = [
        C2CPG_BAT,
        c_filepath,
        "--output", temp_bin
    ]
    
    env = os.environ.copy()
    env["JAVA_OPTS"] = "-Xmx2g -Xms256m"
    
    try:
        res1 = subprocess.run(c2cpg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=120)
        if not os.path.exists(temp_bin) or os.path.getsize(temp_bin) == 0:
            # print(f"Failed c2cpg for {func_id}: {res1.stderr}")
            return False
            
        # 2. Export to DOT format using joern-export.bat
        export_cmd = [
            JOERN_EXPORT_BAT,
            temp_bin,
            "--repr", "all",
            "--format", "dot",
            "--out", temp_export_dir
        ]
        res2 = subprocess.run(export_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, timeout=120)
        
        exported_dot = os.path.join(temp_export_dir, "export.dot")
        if os.path.exists(exported_dot):
            os.makedirs(os.path.dirname(output_dot_path), exist_ok=True)
            shutil.copyfile(exported_dot, output_dot_path)
            return True
        else:
            return False
            
    except Exception as e:
        print(f"Error processing {func_id}: {e}")
        return False
    finally:
        # Clean up temporary files
        if os.path.exists(temp_bin):
            try:
                os.remove(temp_bin)
            except Exception:
                pass
        if os.path.exists(temp_export_dir):
            try:
                shutil.rmtree(temp_export_dir, ignore_errors=True)
            except Exception:
                pass

from concurrent.futures import ThreadPoolExecutor, as_completed

def process_single_c_file(args_tuple):
    c_file, dot_output_dir, temp_dir = args_tuple
    func_id = os.path.splitext(os.path.basename(c_file))[0]
    out_dot = os.path.join(dot_output_dir, f"{func_id}.dot")
    
    # If already generated, skip
    if os.path.exists(out_dot) and os.path.getsize(out_dot) > 0:
        return True, func_id
        
    success = generate_cpg_for_file(c_file, out_dot, temp_dir)
    return success, func_id

def process_split(split_name: str, limit: int = None, workers: int = 4):
    raw_split_dir = os.path.join(PROJECT_ROOT, "data", "raw", "devign", split_name)
    dot_output_dir = os.path.join(PROJECT_ROOT, "data", "graphs", "raw_dot", split_name)
    temp_dir = os.path.join(PROJECT_ROOT, "data", "temp_joern", split_name)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(dot_output_dir, exist_ok=True)
    
    c_files = glob.glob(os.path.join(raw_split_dir, "*.c"))
    if not c_files:
        print(f"No .c files found in {raw_split_dir}")
        return
        
    if limit is not None and limit < len(c_files):
        c_files = c_files[:limit]
        
    print(f"\nProcessing {len(c_files)} files for split '{split_name}' with {workers} workers...")
    
    successful = 0
    failed = 0
    
    tasks = [(c, dot_output_dir, temp_dir) for c in c_files]
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_single_c_file, t): t for t in tasks}
        with tqdm(total=len(tasks), desc=f"Joern CPG [{split_name}]") as pbar:
            for future in as_completed(futures):
                success, func_id = future.result()
                if success:
                    successful += 1
                else:
                    failed += 1
                pbar.update(1)
            
    print(f"Split '{split_name}' Results: {successful} successful, {failed} failed.")
    # Clean up temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Generate Joern CPGs (.dot format) from C source files.")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files per split")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent worker threads (default: 4)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("STEP 4: CODE PROPERTY GRAPH GENERATION VIA JOERN")
    print(f"Joern path: {JOERN_DIR}")
    print("=" * 70)
    
    if not os.path.exists(C2CPG_BAT):
        print(f"Error: Joern executable not found at {C2CPG_BAT}")
        return
        
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    
    for split in splits:
        process_split(split, limit=args.limit, workers=args.workers)
        
    print("\n" + "=" * 70)
    print("CPG generation completed!")
    print("=" * 70)

if __name__ == "__main__":
    main()

