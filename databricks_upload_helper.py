"""
Databricks helper code to upload BLS cache files.
Copy this code into a Databricks notebook cell to upload files.
"""

DATABRICKS_UPLOAD_CODE = """
# Upload BLS cache files to Databricks
# Run this in a Databricks notebook

import os
import shutil
from pathlib import Path

# Option 1: If you uploaded a zip file
def extract_bls_cache_zip(zip_path="bls_cache.zip", extract_to="bls_cache"):
    import zipfile
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        zipf.extractall(extract_to)
    print(f"Extracted {len(list(Path(extract_to).glob('*.csv')))} CSV files to {extract_to}")

# Option 2: Upload files using dbutils (if files are in DBFS)
def upload_files_to_dbfs(local_path, dbfs_path):
    # Copy from local to DBFS
    dbutils.fs.cp(f"file:{local_path}", f"dbfs:{dbfs_path}", recurse=True)
    print(f"Uploaded {local_path} to {dbfs_path}")

# Option 3: List and verify uploaded files
def verify_bls_cache(cache_dir="bls_cache"):
    cache_path = Path(cache_dir)
    if cache_path.exists():
        csv_files = list(cache_path.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in {cache_dir}")
        print(f"States: {', '.join(sorted([f.stem for f in csv_files]))}")
        return csv_files
    else:
        print(f"Cache directory not found: {cache_dir}")
        return []

# Run verification
verify_bls_cache("bls_cache")
"""

if __name__ == "__main__":
    print("This is a helper file for Databricks.")
    print("Copy the code below into a Databricks notebook cell:")
    print("\n" + "=" * 60)
    print(DATABRICKS_UPLOAD_CODE)
    print("=" * 60)

