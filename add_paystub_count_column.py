"""
Add paystub count column corresponding to each hourly rate.

This script reads the CSV file and adds a new column that shows the paystub count
for each hourly rate in the regular_hourly_rate_parsed_counts column.
"""

import pandas as pd
import json
import re
from typing import Dict


def parse_rate_counts(rate_str: str) -> Dict[str, int]:
    """
    Parse hourly rates and their counts from JSON-like string.
    
    Args:
        rate_str: String like '{"25.72": 1}' or '{"17.25": 1, "23.01": 1}'
                 The CSV has escaped quotes like '{""25.72"": 1}'
        
    Returns:
        Dictionary mapping rate strings to counts
    """
    if pd.isna(rate_str) or rate_str == '' or rate_str == '{}':
        return {}
    
    try:
        # The CSV has double quotes escaped as '""', so we need to handle that
        # First, replace '""' with '"' to unescape
        cleaned = rate_str.replace('""', '"')
        # Parse as JSON
        rate_dict = json.loads(cleaned)
        return rate_dict
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        # Try regex fallback for malformed JSON
        # Extract numbers and their counts
        result = {}
        # Pattern to match: "number": count or ""number"": count
        pattern = r'"{1,2}([0-9]+\.?[0-9]*)"{1,2}:\s*([0-9]+)'
        matches = re.findall(pattern, rate_str)
        for rate, count in matches:
            result[rate] = int(count)
        return result


def format_paystub_counts(rate_counts: Dict[str, int]) -> str:
    """
    Format paystub counts in a readable format similar to the rate format.
    
    Args:
        rate_counts: Dictionary mapping rate strings to counts
        
    Returns:
        Formatted string like '{"17.25": 1, "23.01": 1}'
    """
    if not rate_counts:
        return '{}'
    
    # Format as JSON-like string with double quotes
    formatted_items = []
    for rate, count in sorted(rate_counts.items(), key=lambda x: float(x[0])):
        formatted_items.append(f'"{rate}": {count}')
    
    return '{' + ', '.join(formatted_items) + '}'


def add_paystub_count_column(input_csv: str, output_csv: str = None):
    """
    Add paystub count column to CSV file.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file (defaults to input_csv with _updated suffix)
    """
    if output_csv is None:
        # Create output filename by adding _updated before .csv
        if input_csv.endswith('.csv'):
            output_csv = input_csv[:-4] + '_updated.csv'
        else:
            output_csv = input_csv + '_updated.csv'
    
    print(f"Reading CSV file: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows")
    
    # Parse rate counts and create paystub count column
    paystub_counts = []
    
    for idx, row in df.iterrows():
        rate_str = row.get('regular_hourly_rate_parsed_counts', '{}')
        rate_counts = parse_rate_counts(rate_str)
        
        # Format the paystub counts
        formatted_counts = format_paystub_counts(rate_counts)
        paystub_counts.append(formatted_counts)
    
    # Add the new column after regular_hourly_rate_parsed_counts
    # Find the index of regular_hourly_rate_parsed_counts column
    cols = list(df.columns)
    if 'regular_hourly_rate_parsed_counts' in cols:
        rate_col_idx = cols.index('regular_hourly_rate_parsed_counts')
        # Insert after regular_hourly_rate_parsed_counts
        df.insert(rate_col_idx + 1, 'paystub_count_by_rate', paystub_counts)
    else:
        # If column not found, append at the end
        df['paystub_count_by_rate'] = paystub_counts
    
    # Save to output file
    print(f"Saving updated CSV to: {output_csv}")
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved {len(df)} rows to {output_csv}")
    
    return df


if __name__ == "__main__":
    import sys
    
    # Default file path
    input_csv = "SEO - Sheet2.csv"
    
    # Allow command line argument
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    
    output_csv = None
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    
    try:
        add_paystub_count_column(input_csv, output_csv)
        print("\n✓ Processing complete!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

