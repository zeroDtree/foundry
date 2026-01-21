"""
Inspect Parquet files to see their structure
"""

import sys
import pandas as pd
from pathlib import Path


def inspect_parquet(parquet_path: str):
    """Inspect a Parquet file and display its structure."""
    
    if not Path(parquet_path).exists():
        print(f"Error: File not found: {parquet_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"Inspecting: {parquet_path}")
    print(f"{'='*70}")
    
    # Read the Parquet file
    df = pd.read_parquet(parquet_path)
    
    # Basic info
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column information
    print(f"\n{'Columns:':<30} {'Type':<20} {'Non-Null':<15} {'Unique'}")
    print("-" * 70)
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = f"{df[col].notna().sum():,}"
        unique = f"{df[col].nunique():,}"
        print(f"{col:<30} {dtype:<20} {non_null:<15} {unique}")
    
    # Metadata (if exists)
    if hasattr(df, 'attrs') and df.attrs:
        print(f"\nMetadata (attrs):")
        for key, value in df.attrs.items():
            print(f"  {key}: {value}")
    else:
        print(f"\nNo metadata found.")
    
    # Sample data
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_parquet.py <parquet_file1> [parquet_file2] ...")
        print("\nExample:")
        print("  python inspect_parquet.py data/pocketgen_parquet/interfaces_df.parquet")
        sys.exit(1)
    
    for parquet_path in sys.argv[1:]:
        inspect_parquet(parquet_path)
