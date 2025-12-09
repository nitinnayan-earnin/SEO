"""
Create a zip file of the BLS cache for easier upload to Databricks.
"""

import os
import zipfile
import sys
from pathlib import Path

def create_zip(cache_dir="bls_cache", output_zip="bls_cache.zip"):
    """
    Create a zip file containing all CSV files from the cache directory.
    
    Args:
        cache_dir: Source cache directory
        output_zip: Output zip file name
    """
    cache_dir = Path(cache_dir).resolve()
    
    if not cache_dir.exists():
        print(f"Error: Cache directory not found: {cache_dir}")
        return False
    
    csv_files = list(cache_dir.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {cache_dir}")
        return False
    
    print(f"Found {len(csv_files)} CSV files")
    print(f"Creating zip file: {output_zip}...")
    
    total_size = 0
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for idx, csv_file in enumerate(sorted(csv_files), 1):
            file_size = csv_file.stat().st_size
            total_size += file_size
            zipf.write(csv_file, csv_file.name)
            if idx % 10 == 0:
                print(f"  Added {idx}/{len(csv_files)} files...")
    
    zip_size = Path(output_zip).stat().st_size
    print(f"\n✓ Created {output_zip}")
    print(f"  Total files: {len(csv_files)}")
    print(f"  Uncompressed size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
    print(f"  Compressed size: {zip_size:,} bytes ({zip_size/1024/1024:.2f} MB)")
    print(f"\nTo upload to Databricks:")
    print(f"  1. Upload {output_zip} to Databricks")
    print(f"  2. In Databricks notebook, run:")
    print(f"     import zipfile")
    print(f"     import os")
    print(f"     os.makedirs('bls_cache', exist_ok=True)")
    print(f"     with zipfile.ZipFile('{output_zip}', 'r') as zipf:")
    print(f"         zipf.extractall('bls_cache')")
    
    return True


if __name__ == "__main__":
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "bls_cache"
    output_zip = sys.argv[2] if len(sys.argv) > 2 else "bls_cache.zip"
    
    create_zip(cache_dir, output_zip)

