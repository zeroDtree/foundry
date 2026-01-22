mkdir -p data
uv run atomworks setup metadata data/metadata
uv run atomworks pdb sync data/pdb_data --pdb-id 1A0I --pdb-id 7XYZ
ln -s $(pwd)/data/metadata/pn_units_df.parquet data/metadata/pn_units_df_train.parquet