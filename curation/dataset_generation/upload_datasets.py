"""Upload one or more CSV datasets to VIS and optionally register them in MLP.

Template commands:

1) Single CSV (dataset name inferred from filename):
    python scripts/curation/upload_datasets.py path/to/data.csv

2) Single CSV with explicit dataset name:
    python scripts/curation/upload_datasets.py \
         path/to/data.csv \
         --dataset-names my_custom_dataset_name

3) Multiple CSVs with explicit names (order must match):
    python scripts/curation/upload_datasets.py \
         path/to/a.csv path/to/b.csv \
         --dataset-names dataset_a dataset_b

4) VIS upload only (skip MLP registration):
    python scripts/curation/upload_datasets.py \
         path/to/data.csv \
         --skip-mlp

5) Custom image column and dataset type:
    python scripts/curation/upload_datasets.py \
         path/to/data.csv \
         --image-col im_url \
         --type DETECTION
"""

import os
import argparse
import pandas as pd
from data_factory.parser.vis_upload import VISUploader
import data_factory.client.hydravision as hv

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic script to upload a collection of CSV datasets to VIS and MLP console."
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="One or more paths to the CSV files to process."
    )
    parser.add_argument(
        "--image-col", "-c",
        default="im_url",
        help="The column name containing the source image URLs/paths (default: 'im_url')."
    )
    parser.add_argument(
        "--type", "-t",
        choices=["DETECTION", "CLASSIFICATION", "IMAGE_ONLY"],
        default="DETECTION",
        help="The MLP dataset type (default: 'DETECTION')."
    )
    parser.add_argument(
        "--remove-failures",
        action="store_true",
        help="If set, rows with failed image downloads will be removed from the final CSV. By default, failed rows are kept."
    )
    parser.add_argument(
        "--skip-mlp",
        action="store_true",
        help="If set, only upload images to VIS and update CSVs locally, skipping MLP dataset registration."
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=None,
        help=(
            "Optional dataset names to use in MLP instead of CSV filenames. "
            "If provided, count must match number of csv_files."
        ),
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.dataset_names is not None and len(args.dataset_names) != len(args.csv_files):
        print(
            "Error: --dataset-names count must match number of csv_files. "
            f"Got {len(args.dataset_names)} names for {len(args.csv_files)} files."
        )
        return

    # 1. Load dataframes and map path to name
    dfs = {}
    files = {}
    for i, path in enumerate(args.csv_files):
        if not os.path.exists(path):
            print(f"Error: File {path} not found.")
            return
        
        # Infer dataset name from CSV filename unless overridden.
        dataset_name = (
            args.dataset_names[i]
            if args.dataset_names is not None
            else os.path.splitext(os.path.basename(path))[0]
        )
        if dataset_name in dfs:
            print(f"Error: Duplicate dataset name '{dataset_name}'. Use unique names.")
            return
        print(f"Loading '{path}' as dataset '{dataset_name}'...")
        dfs[dataset_name] = pd.read_csv(path)
        files[dataset_name] = path

    # 2. Consolidate all unique image URLs/paths to upload to VIS in one batch
    print("\nConsolidating unique URLs for VIS upload...")
    all_urls_series = []
    for name, df in dfs.items():
        if args.image_col not in df.columns:
            print(f"Error: Column '{args.image_col}' not found in '{files[name]}'. Available: {list(df.columns)}")
            return
        all_urls_series.append(df[args.image_col])
    
    all_urls = pd.concat(all_urls_series).unique()
    url_df = pd.DataFrame({args.image_col: all_urls})
    print(f"Total unique images to check/upload: {len(all_urls)}")

    # Initialize VISUploader
    uploader = VISUploader(
        input_url_column_name=args.image_col,
        vis_url_column_name=args.image_col,
        remove_download_fail=args.remove_failures,
        use_prev_session=True
    )

    # Perform VIS upload
    processed_url_df = uploader.process_dataframes([url_df])[0]

    # Map original web URLs to their generated vis_uri and hosted im_url
    # Under the hood, VISUploader renames the input column to 'original_url' and creates the target column
    url_map = processed_url_df.set_index('original_url')[['vis_uri', args.image_col]].to_dict('index')

    # 3. Update original dataframes and save updated CSVs
    print("\nUpdating CSV files with VIS details...")
    updated_dfs = {}
    for name, df in dfs.items():
        # Map details back to original dataframe columns
        df['vis_uri'] = df[args.image_col].map(lambda x: url_map[x]['vis_uri'] if x in url_map else '')
        df['original_url'] = df[args.image_col]
        df[args.image_col] = df[args.image_col].map(lambda x: url_map[x][args.image_col] if x in url_map else x)

        # Handle rows where image upload failed
        if args.remove_failures:
            valid_df = df[df['vis_uri'] != ''].copy()
        else:
            valid_df = df.copy()

        successful_uploads = valid_df['vis_uri'].ne('').sum()
        print(f"Dataset '{name}': {successful_uploads} / {len(valid_df)} images successfully uploaded to VIS.")

        # Overwrite the original CSV file with updated columns
        valid_df.to_csv(files[name], index=False)
        updated_dfs[name] = valid_df

    # 4. Create and upload the datasets to MLP console
    if not args.skip_mlp:
        print("\nRegistering and uploading datasets to MLP (HydraVision)...")
        for name, df in updated_dfs.items():
            print(f"Uploading dataset '{name}' to MLP...")
            hv_dataset = hv.HydraVisionCreateDataset(
                dataset_name=name,
                dataset_type=args.type,
                update_existing=True
            )
            hv_dataset.write_dataframe(df)
            print(f"Successfully uploaded dataset '{name}' to MLP.")
    else:
        print("\nSkipped MLP registration (--skip-mlp was set).")

    print("\nAll processing completed successfully!")

if __name__ == "__main__":
    main()
