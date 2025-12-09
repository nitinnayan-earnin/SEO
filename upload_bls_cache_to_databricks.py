"""
Upload BLS cache files to Databricks.
This script helps upload files individually to avoid UI limitations.
"""

import os
import subprocess
import sys
from pathlib import Path

def upload_via_cli(cache_dir="bls_cache", databricks_path="/Workspace/Users/nitin.nayan@earnin.com/SEO/bls_cache"):
    """
    Upload cache files using Databricks CLI.
    
    Args:
        cache_dir: Local cache directory
        databricks_path: Target path in Databricks
    """
    cache_dir = Path(cache_dir).resolve()
    
    if not cache_dir.exists():
        print(f"Error: Cache directory not found: {cache_dir}")
        return False
    
    csv_files = list(cache_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {cache_dir}")
        return False
    
    print(f"Found {len(csv_files)} CSV files to upload")
    print(f"Target: {databricks_path}")
    
    # Check if databricks CLI is available
    try:
        result = subprocess.run(["databricks", "--version"], capture_output=True, text=True)
        print(f"Databricks CLI version: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Error: Databricks CLI not found. Install it first:")
        print("  pip install databricks-cli")
        return False
    
    successful = []
    failed = []
    
    for idx, csv_file in enumerate(sorted(csv_files), 1):
        state_code = csv_file.stem
        target_file = f"{databricks_path}/{state_code}.csv"
        
        print(f"[{idx}/{len(csv_files)}] Uploading {state_code}.csv ({csv_file.stat().st_size:,} bytes)...", end=" ", flush=True)
        
        try:
            # Upload file using databricks CLI
            cmd = [
                "databricks", "fs", "cp",
                str(csv_file),
                f"dbfs:{target_file}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✓")
                successful.append(state_code)
            else:
                print(f"✗ Error: {result.stderr}")
                failed.append(state_code)
                
        except subprocess.TimeoutExpired:
            print("✗ Timeout")
            failed.append(state_code)
        except Exception as e:
            print(f"✗ Error: {e}")
            failed.append(state_code)
    
    # Summary
    print("\n" + "=" * 60)
    print("UPLOAD SUMMARY")
    print("=" * 60)
    print(f"Total files: {len(csv_files)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessfully uploaded: {', '.join(successful)}")
    
    if failed:
        print(f"\nFailed to upload: {', '.join(failed)}")
        print("\nYou can try uploading failed files manually or re-run this script.")
    
    return len(failed) == 0


def create_upload_instructions(cache_dir="bls_cache"):
    """
    Create instructions file for manual upload.
    """
    cache_dir = Path(cache_dir).resolve()
    csv_files = sorted(cache_dir.glob("*.csv"))
    
    instructions = []
    instructions.append("=" * 60)
    instructions.append("MANUAL UPLOAD INSTRUCTIONS FOR DATABRICKS")
    instructions.append("=" * 60)
    instructions.append(f"\nTotal files to upload: {len(csv_files)}")
    instructions.append(f"Cache directory: {cache_dir}")
    instructions.append("\nMETHOD 1: Upload via Databricks UI (Recommended for small batches)")
    instructions.append("- Go to Databricks workspace")
    instructions.append("- Navigate to: /Workspace/Users/nitin.nayan@earnin.com/SEO/")
    instructions.append("- Create 'bls_cache' folder if it doesn't exist")
    instructions.append("- Upload files in batches of 10-20 files at a time")
    instructions.append("\nMETHOD 2: Use Databricks CLI")
    instructions.append("Run: python upload_bls_cache_to_databricks.py")
    instructions.append("\nMETHOD 3: Upload individual files")
    instructions.append("For each file, use Databricks UI or CLI:")
    
    for csv_file in csv_files:
        state_code = csv_file.stem
        file_size = csv_file.stat().st_size
        instructions.append(f"  - {state_code}.csv ({file_size:,} bytes)")
    
    instructions.append("\n" + "=" * 60)
    
    instructions_file = cache_dir / "UPLOAD_INSTRUCTIONS.txt"
    with open(instructions_file, 'w') as f:
        f.write('\n'.join(instructions))
    
    print(f"Created upload instructions: {instructions_file}")
    return instructions_file


def verify_upload(cache_dir="bls_cache", databricks_path="/Workspace/Users/nitin.nayan@earnin.com/SEO/bls_cache"):
    """
    Verify which files are already uploaded in Databricks.
    """
    cache_dir = Path(cache_dir).resolve()
    local_files = {f.stem for f in cache_dir.glob("*.csv")}
    
    print(f"Local files: {len(local_files)}")
    print(f"Checking Databricks: {databricks_path}")
    
    try:
        # List files in Databricks
        cmd = [
            "databricks", "fs", "ls",
            f"dbfs:{databricks_path}"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            uploaded_files = set()
            for line in result.stdout.split('\n'):
                if line.strip() and line.endswith('.csv'):
                    filename = line.split()[-1]
                    uploaded_files.add(filename.replace('.csv', ''))
            
            print(f"Uploaded files: {len(uploaded_files)}")
            missing = local_files - uploaded_files
            if missing:
                print(f"\nMissing files ({len(missing)}): {', '.join(sorted(missing))}")
            else:
                print("\n✓ All files are uploaded!")
            return missing
        else:
            print(f"Error listing files: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("Databricks CLI not found. Cannot verify upload.")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload BLS cache to Databricks")
    parser.add_argument("--cache-dir", default="bls_cache", help="Local cache directory")
    parser.add_argument("--databricks-path", default="/Workspace/Users/nitin.nayan@earnin.com/SEO/bls_cache",
                       help="Target path in Databricks")
    parser.add_argument("--verify", action="store_true", help="Verify uploaded files")
    parser.add_argument("--instructions", action="store_true", help="Create upload instructions file")
    
    args = parser.parse_args()
    
    if args.instructions:
        create_upload_instructions(args.cache_dir)
    elif args.verify:
        verify_upload(args.cache_dir, args.databricks_path)
    else:
        success = upload_via_cli(args.cache_dir, args.databricks_path)
        if success:
            print("\n✓ All files uploaded successfully!")
        else:
            print("\n⚠ Some files failed to upload. Check the summary above.")
            print("You can use --verify to check which files are missing.")

