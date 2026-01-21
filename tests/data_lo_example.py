"""
Simple examples showing how to use the dataset loading functionality
"""

from data_lo import (
    load_dataset_from_files,
    load_dataset_from_parquet,
    create_parquet_index,
)


def example_1_direct_loading():
    """Example 1: Direct file loading (simple but slower)"""
    print("\n" + "=" * 70)
    print("Example 1: Direct File Loading")
    print("=" * 70)

    # Load dataset directly from files
    dataset = load_dataset_from_files(
        base_directory="data/pocket",
        file_filter="pocket_pdb",  # Only load *_pocket10.pdb files
        max_depth=2,
    )

    # Access dataset
    print(f"\nDataset size: {len(dataset)}")

    # Get some example IDs
    print("\nFirst 5 examples:")
    for i in range(min(5, len(dataset))):
        example_id = dataset.idx_to_id(i)
        print(f"  {i}: {example_id}")


def example_2_parquet_loading():
    """Example 2: Load from Parquet index (fast!)"""
    print("\n" + "=" * 70)
    print("Example 2: Parquet Index Loading")
    print("=" * 70)

    # Create index if it doesn't exist
    import os

    parquet_path = "data/pocket_index.parquet"
    if not os.path.exists(parquet_path):
        print("\nCreating Parquet index (one-time setup)...")
        create_parquet_index(
            base_directory="data/pocket",
            output_path=parquet_path,
            file_type="pocket_pdb",
        )

    # Load from Parquet (very fast!)
    dataset, df = load_dataset_from_parquet(parquet_path)

    print(f"\nDataset size: {len(dataset)}")

    # Access metadata
    print("\nDataset metadata:")
    for key, value in df.attrs.items():
        print(f"  {key}: {value}")

    # Use the DataFrame for analysis
    print("\nTop 5 proteins by sample count:")
    protein_counts = df["protein_name"].value_counts().head()
    for protein, count in protein_counts.items():
        print(f"  {protein}: {count} samples")


def example_3_dataframe_filtering():
    """Example 3: Filter data using pandas DataFrame"""
    print("\n" + "=" * 70)
    print("Example 3: DataFrame Filtering")
    print("=" * 70)

    # Load dataset with DataFrame
    dataset, df = load_dataset_from_parquet("data/pocket_index.parquet")

    # Filter by protein name
    target_protein = df["protein_name"].iloc[0]  # Get first protein name
    filtered_df = df[df["protein_name"] == target_protein]

    print(f"\nFiltering for protein: {target_protein}")
    print(f"Total samples: {len(df)}")
    print(f"Filtered samples: {len(filtered_df)}")

    # Show filtered examples
    print(f"\nExamples for {target_protein}:")
    for idx, row in filtered_df.head(3).iterrows():
        print(f"  - {row['example_id']}")


def example_4_pytorch_integration():
    """Example 4: Use with PyTorch DataLoader"""
    print("\n" + "=" * 70)
    print("Example 4: PyTorch DataLoader Integration")
    print("=" * 70)

    try:
        from torch.utils.data import DataLoader

        # Load dataset
        dataset, _ = load_dataset_from_parquet("data/pocket_index.parquet")

        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Use 0 for testing, increase for training
        )

        print(f"\nDataLoader created with batch_size=4")
        print(f"Total batches: {len(dataloader)}")

        # Get first batch
        print("\nFirst batch example IDs:")
        batch = next(iter(dataloader))
        # Note: batch will contain the actual data, here we just show it exists
        print(f"  Batch type: {type(batch)}")

    except ImportError:
        print("\nPyTorch not installed. Skipping this example.")


def example_5_custom_index():
    """Example 5: Create custom filtered index"""
    print("\n" + "=" * 70)
    print("Example 5: Custom Filtered Index")
    print("=" * 70)

    import pandas as pd

    # Load full index
    df = pd.read_parquet("data/pocket_index.parquet")

    # Create filtered index (example: only first 100 proteins alphabetically)
    unique_proteins = sorted(df["protein_name"].unique())[:100]
    filtered_df = df[df["protein_name"].isin(unique_proteins)]

    # Save as new index
    custom_path = "data/pocket_index_small.parquet"
    filtered_df.attrs = df.attrs.copy()  # Preserve metadata
    filtered_df.attrs["dataset_name"] = "pocketgen_small_subset"
    filtered_df.attrs["protein_count"] = len(unique_proteins)

    from atomworks.ml.utils.io import to_parquet_with_metadata

    to_parquet_with_metadata(filtered_df, custom_path)

    print(f"\nCreated custom index: {custom_path}")
    print(f"Original size: {len(df)}")
    print(f"Filtered size: {len(filtered_df)}")
    print(f"Proteins included: {len(unique_proteins)}")

    # Load and use the custom index
    from atomworks.ml.datasets import PandasDataset

    custom_dataset = PandasDataset(
        data=custom_path, id_column="example_id", name="pocketgen_small"
    )
    print(f"\nCustom dataset loaded: {len(custom_dataset)} examples")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PocketGen Dataset Loading Examples")
    print("=" * 70)

    # Run examples
    # Note: Comment out examples you don't want to run

    # Example 1: Direct file loading (slower, no preprocessing needed)
    # example_1_direct_loading()

    # Example 2: Parquet loading (fast, requires one-time index creation)
    example_2_parquet_loading()

    # Example 3: Filter data using DataFrame
    # example_3_dataframe_filtering()

    # Example 4: PyTorch DataLoader integration
    # example_4_pytorch_integration()

    # Example 5: Create custom filtered index
    # example_5_custom_index()

    print("\n" + "=" * 70)
    print("Examples completed!")
    print("=" * 70)
