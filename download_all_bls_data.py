"""
Download all BLS state data and save to cache directory.
Run this locally (not in Databricks) to pre-populate the cache.
Then upload the bls_cache directory to Databricks.
"""

import os
import sys
from state_links import BLSOESDataExtractor

def download_all_states(cache_dir="bls_cache"):
    """
    Download BLS data for all states and save to cache directory.
    
    Args:
        cache_dir: Directory to save cached state data files
    """
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    cache_dir = os.path.abspath(cache_dir)
    print(f"Cache directory: {cache_dir}")
    
    # Initialize extractor
    print("Initializing BLS data extractor...")
    extractor = BLSOESDataExtractor(headless=True, verbose=True)
    
    # Get all state codes
    all_states = sorted(extractor.STATE_LINKS.keys())
    print(f"Found {len(all_states)} states to download")
    
    successful = []
    failed = []
    skipped = []
    
    for idx, state_code in enumerate(all_states, 1):
        cache_file = os.path.join(cache_dir, f"{state_code}.csv")
        
        # Skip if already cached
        if os.path.exists(cache_file):
            print(f"[{idx}/{len(all_states)}] {state_code}: Already cached, skipping...")
            skipped.append(state_code)
            continue
        
        print(f"[{idx}/{len(all_states)}] Downloading {state_code}...")
        try:
            df = extractor.get_state_data(state_code, clean_data=True)
            
            if df is not None and not df.empty:
                df.to_csv(cache_file, index=False)
                file_size = os.path.getsize(cache_file)
                print(f"  ✓ Successfully cached {len(df)} rows ({file_size:,} bytes)")
                successful.append(state_code)
            else:
                print(f"  ✗ No data returned for {state_code}")
                failed.append(state_code)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed.append(state_code)
    
    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total states: {len(all_states)}")
    print(f"Successful: {len(successful)}")
    print(f"Skipped (already cached): {len(skipped)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessfully cached states: {', '.join(successful)}")
    
    if failed:
        print(f"\nFailed states: {', '.join(failed)}")
    
    if skipped:
        print(f"\nSkipped (already cached): {', '.join(skipped)}")
    
    print(f"\nCache directory: {cache_dir}")
    print(f"To use in Databricks, upload the '{cache_dir}' directory to your Databricks workspace.")
    print("=" * 60)


if __name__ == "__main__":
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else "bls_cache"
    download_all_states(cache_dir)

