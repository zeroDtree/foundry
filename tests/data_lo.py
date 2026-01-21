"""
PocketGen Dataset Loader with Parquet Support

This script provides two ways to load the PocketGen dataset:
1. Direct file scanning (FileDataset) - Simple but slower for large datasets
2. Parquet index (PandasDataset) - Fast and supports metadata

Usage:
    # Method 1: Direct file scanning
    python tests/data_lo.py --method file

    # Method 2: Create and use Parquet index
    python tests/data_lo.py --method parquet --create-index

    # Method 3: Use existing Parquet index
    python tests/data_lo.py --method parquet
"""

import argparse
from pathlib import Path
from typing import Literal

import pandas as pd
from atomworks.ml.datasets import FileDataset, PandasDataset
from atomworks.ml.utils.io import to_parquet_with_metadata, read_parquet_with_metadata


class CustomFileDataset(FileDataset):
    """Custom FileDataset that uses relative path as example ID."""

    def __init__(self, base_dir: str, *args, **kwargs):
        self.base_dir = Path(base_dir)
        super().__init__(*args, **kwargs)

    def _get_example_id(self, idx: int) -> str:
        """Override to use relative path as ID instead of just filename."""
        file_path = self.file_paths[idx]
        # Get relative path from base directory
        try:
            rel_path = file_path.relative_to(self.base_dir)
        except ValueError:
            # If file is not relative to base_dir, use absolute path
            rel_path = file_path

        # Remove extension and return
        id_str = str(rel_path)
        for ext in [".pdb", ".sdf", ".cif", ".gz"]:
            id_str = id_str.replace(ext, "")
        return id_str


# ============================================================================
# Filter Functions
# ============================================================================


def pdb_only_filter(filepath) -> bool:
    """Only include .pdb files."""
    return str(filepath).endswith(".pdb")


def pocket_pdb_only_filter(filepath) -> bool:
    """Only include _pocket10.pdb files."""
    return str(filepath).endswith("_pocket10.pdb")


def sdf_only_filter(filepath) -> bool:
    """Only include .sdf files."""
    return str(filepath).endswith(".sdf")


# ============================================================================
# Parquet Index Creation
# ============================================================================


def create_parquet_index(
    base_directory: str,
    output_path: str = "data/pocketgen_parquet/index.parquet",
    file_type: Literal["pdb", "pocket_pdb", "sdf", "all"] = "pocket_pdb",
    max_depth: int = 2,
) -> pd.DataFrame:
    """
    Create a Parquet index file for the PocketGen dataset.

    Args:
        base_directory: Root directory containing the pocket data
        output_path: Path to save the Parquet index file
        file_type: Type of files to index ('pdb', 'pocket_pdb', 'sdf', 'all')
        max_depth: Maximum directory depth to search

    Returns:
        pandas DataFrame containing the indexed files
    """
    print(f"Creating Parquet index for {base_directory}...")
    print(f"File type filter: {file_type}")

    base_path = Path(base_directory)
    file_records = []

    # Define file patterns based on file_type
    if file_type == "pocket_pdb":
        patterns = ["*_pocket10.pdb"]
    elif file_type == "pdb":
        patterns = ["*.pdb"]
    elif file_type == "sdf":
        patterns = ["*.sdf"]
    elif file_type == "all":
        patterns = ["*.pdb", "*.sdf"]
    else:
        raise ValueError(f"Unknown file_type: {file_type}")

    # Scan for files
    for pattern in patterns:
        if max_depth == 1:
            files = base_path.glob(pattern)
        else:
            files = base_path.rglob(pattern)

        for file_path in files:
            rel_path = file_path.relative_to(base_path)

            # Extract metadata from path and filename
            parts = rel_path.parts
            protein_name = parts[0] if len(parts) > 1 else "unknown"
            filename_stem = file_path.stem

            # Parse filename components (e.g., "1abc_A_rec_1def_lig_tt_docked_0_pocket10")
            name_parts = filename_stem.split("_")

            record = {
                "example_id": str(rel_path).replace(".pdb", "").replace(".sdf", ""),
                "file_path": str(file_path),
                "relative_path": str(rel_path),
                "protein_name": protein_name,
                "filename": file_path.name,
                "file_type": file_path.suffix[1:],  # Remove the dot
            }

            # Try to extract receptor and ligand info from filename
            if len(name_parts) >= 5:
                record["receptor"] = name_parts[0]
                record["ligand"] = name_parts[4] if len(name_parts) > 4 else None

            # Check if it's a pocket file
            record["is_pocket"] = "_pocket10" in filename_stem

            file_records.append(record)

    # Create DataFrame
    df = pd.DataFrame(file_records)

    # Add metadata to DataFrame
    df.attrs["dataset_name"] = "pocketgen_crossdocked"
    df.attrs["base_directory"] = str(base_directory)
    df.attrs["file_type"] = file_type
    df.attrs["total_files"] = len(df)
    df.attrs["created_at"] = pd.Timestamp.now().isoformat()

    # Save to Parquet
    print(f"Saving {len(df)} records to {output_path}...")
    to_parquet_with_metadata(df, output_path)

    # Print statistics
    print(f"\nIndex created successfully!")
    print(f"Total files: {len(df)}")
    print(f"Unique proteins: {df['protein_name'].nunique()}")
    if "is_pocket" in df.columns:
        print(f"Pocket files: {df['is_pocket'].sum()}")
    print(f"File types: {df['file_type'].value_counts().to_dict()}")

    return df


# ============================================================================
# Dataset Loading Functions
# ============================================================================


def load_dataset_from_files(
    base_directory: str = "data/pocket",
    file_filter: Literal["pdb", "pocket_pdb", "sdf"] = "pocket_pdb",
    max_depth: int = 2,
) -> CustomFileDataset:
    """
    Load dataset by directly scanning files.

    Args:
        base_directory: Root directory containing the data
        file_filter: Type of files to load
        max_depth: Maximum directory depth to search

    Returns:
        CustomFileDataset instance
    """
    print(f"\nLoading dataset from files in {base_directory}...")

    # Select filter function
    filter_fn_map = {
        "pdb": pdb_only_filter,
        "pocket_pdb": pocket_pdb_only_filter,
        "sdf": sdf_only_filter,
    }
    filter_fn = filter_fn_map.get(file_filter)

    dataset = CustomFileDataset.from_directory(
        directory=base_directory,
        name="pocketgen_data",
        max_depth=max_depth,
        base_dir=base_directory,
        filter_fn=filter_fn,
    )

    print(f"✓ Loaded {len(dataset)} examples")
    return dataset


def load_dataset_from_parquet(
    parquet_path: str = "data/pocket_index.parquet",
) -> tuple[PandasDataset, pd.DataFrame]:
    """
    Load dataset from a Parquet index file.

    Args:
        parquet_path: Path to the Parquet index file

    Returns:
        Tuple of (PandasDataset instance, DataFrame with metadata)
    """
    print(f"\nLoading dataset from Parquet index: {parquet_path}...")

    # Read Parquet with metadata
    df = read_parquet_with_metadata(parquet_path)

    # Print metadata
    print(f"Dataset metadata:")
    for key, value in df.attrs.items():
        print(f"  {key}: {value}")

    # Create PandasDataset
    dataset = PandasDataset(
        data=parquet_path,
        id_column="example_id",
        name="pocketgen_data",
    )

    print(f"✓ Loaded {len(dataset)} examples")
    return dataset, df


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Load PocketGen dataset with optional Parquet indexing"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["file", "parquet"],
        default="file",
        help="Method to load dataset: 'file' (direct scan) or 'parquet' (use index)",
    )
    parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create Parquet index before loading (only with --method parquet)",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="data/pocket",
        help="Base directory containing pocket data",
    )
    parser.add_argument(
        "--parquet-path",
        type=str,
        default="data/pocketgen_parquet/index.parquet",
        help="Path to Parquet index file",
    )
    parser.add_argument(
        "--file-type",
        type=str,
        choices=["pdb", "pocket_pdb", "sdf", "all"],
        default="pocket_pdb",
        help="Type of files to load/index",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum directory depth to search",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=5,
        help="Number of example IDs to display",
    )

    args = parser.parse_args()

    # ========================================================================
    # Load or Create Dataset
    # ========================================================================

    if args.method == "file":
        # Method 1: Direct file scanning
        dataset = load_dataset_from_files(
            base_directory=args.base_dir,
            file_filter=args.file_type if args.file_type != "all" else "pdb",
            max_depth=args.max_depth,
        )
        df = None

    elif args.method == "parquet":
        # Method 2: Use Parquet index
        if args.create_index or not Path(args.parquet_path).exists():
            # Create index if requested or if it doesn't exist
            df = create_parquet_index(
                base_directory=args.base_dir,
                output_path=args.parquet_path,
                file_type=args.file_type,
                max_depth=args.max_depth,
            )

        # Load from Parquet
        dataset, df = load_dataset_from_parquet(parquet_path=args.parquet_path)

    # ========================================================================
    # Display Dataset Information
    # ========================================================================

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Total examples: {len(dataset)}")

    # Show example IDs
    print(f"\nFirst {args.show_examples} example IDs:")
    for i in range(min(args.show_examples, len(dataset))):
        example_id = dataset.idx_to_id(i)
        print(f"  {i + 1:3d}. {example_id}")

    # If we have DataFrame, show additional statistics
    if df is not None and args.method == "parquet":
        print("\nDataset Statistics:")
        print(f"  Unique proteins: {df['protein_name'].nunique()}")
        print(f"  File types: {df['file_type'].value_counts().to_dict()}")
        if "is_pocket" in df.columns:
            print(f"  Pocket files: {df['is_pocket'].sum()}")
            print(f"  Non-pocket files: {(~df['is_pocket']).sum()}")

        # Show sample of the DataFrame
        print("\nSample records from index:")
        print(df[["example_id", "protein_name", "file_type", "is_pocket"]].head(3))

    print("=" * 70)


if __name__ == "__main__":
    main()
