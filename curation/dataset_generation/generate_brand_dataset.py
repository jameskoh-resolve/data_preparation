#!/usr/bin/env python
"""
Generic dataset generation script for any brand.

Replaces per-brand generate_<brand>_dataset.py scripts. Finds the next version
number and runs scripts/curation/generate_datasets.py to clean, verify,
distribute, and split the data into versioned directories.

Usage:
    python scripts/curation/generate_brand_dataset.py --brand asos [--input-file curated_issues_final.csv] [--skip-upload]
    python scripts/curation/generate_brand_dataset.py --brand veromoda --base-dir curated_datasets/Veromoda_India
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path


def find_next_version(brand_dir):
    """Find the next version number by scanning v* directories."""
    max_version = 0
    for p in brand_dir.iterdir():
        if p.is_dir() and re.match(r"^v\d+$", p.name):
            try:
                v = int(p.name[1:])
                if v > max_version:
                    max_version = v
            except ValueError:
                pass
    return max_version + 1


def resolve_input_file(input_file, curation_dir):
    """Resolve the input file path, checking multiple locations."""
    input_path = Path(input_file)
    if input_path.is_absolute() and input_path.exists():
        return input_path.resolve()

    # Check relative to cwd
    if input_path.exists():
        return input_path.resolve()

    # Check relative to curation dir
    candidate = curation_dir / input_file
    if candidate.exists():
        return candidate.resolve()

    # Check in corrections subdirectory
    candidate = curation_dir / "corrections" / input_file
    if candidate.exists():
        return candidate.resolve()

    return None


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    generate_datasets_script = project_root / "curation" / "dataset_generation" / "generate_datasets.py"
    upload_script = project_root / "curation" / "dataset_generation" / "upload_datasets.py"

    parser = argparse.ArgumentParser(description="Generate the next version of a brand dataset.")
    parser.add_argument("--brand", required=True, help="Brand name (e.g. asos, cos, veromoda, selectedhomme).")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Base directory for the brand (default: curated_datasets/<Brand>/)."
    )
    parser.add_argument(
        "--input-file",
        default="curated_issues_final.csv",
        help="Path to the input curated CSV (default: 'curated_issues_final.csv')."
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip uploading the generated datasets to VIS and MLP."
    )
    args = parser.parse_args()

    brand = args.brand.lower()

    # Resolve base directory
    if args.base_dir:
        brand_dir = Path(args.base_dir).resolve()
    else:
        # Try to find the brand directory in curated_datasets/
        curated_root = project_root / "curated_datasets"
        # Look for exact match or case-insensitive match
        brand_dir = None
        for p in curated_root.iterdir():
            if p.is_dir() and p.name.lower().replace("_", "").replace("-", "") == brand.replace("_", "").replace("-", ""):
                brand_dir = p
                break
        if brand_dir is None:
            brand_dir = curated_root / brand.capitalize()

    curation_dir = brand_dir / "curation"
    if not curation_dir.exists():
        print(f"Error: Curation directory does not exist: {curation_dir}")
        print(f"Run scaffold_brand.py first to set up the directory structure.")
        sys.exit(1)

    # Resolve input file
    input_path = resolve_input_file(args.input_file, curation_dir)
    if input_path is None:
        print(f"Error: Input file not found: {args.input_file}")
        print(f"Searched in: cwd, {curation_dir}, {curation_dir / 'corrections'}")
        sys.exit(1)

    # Find the next version number
    next_version = find_next_version(brand_dir)
    next_dir = brand_dir / f"v{next_version}"
    next_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running dataset generation pipeline...")
    print(f"Brand:  {brand}")
    print(f"Input:  {input_path}")
    print(f"Output: {next_dir}")

    # Run the curation pipeline
    cmd = [
        sys.executable,
        str(generate_datasets_script),
        str(input_path),
        str(next_dir),
        brand,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: generate_datasets.py failed with exit code {e.returncode}")
        sys.exit(e.returncode)

    # Rename output files to follow versioned naming convention
    # e.g., asos_dataset.csv -> asos_dataset_v3.csv
    prefix = f"{brand}_dataset"
    print(f"\nRenaming generated files to follow versioned pattern...")
    for p in next_dir.iterdir():
        if p.name.startswith(prefix):
            new_name = p.name.replace(prefix, f"{prefix}_v{next_version}", 1)
            new_path = next_dir / new_name
            p.rename(new_path)
            print(f"  {p.name} -> {new_name}")

    print(f"\nSuccessfully generated version {next_version} dataset in {next_dir}!")

    if not args.skip_upload:
        csv_files = [
            str(next_dir / f"{prefix}_v{next_version}.csv"),
            str(next_dir / f"{prefix}_v{next_version}_train.csv"),
            str(next_dir / f"{prefix}_v{next_version}_val.csv"),
            str(next_dir / f"{prefix}_v{next_version}_test.csv"),
        ]

        # Filter to only existing files
        existing_csv_files = [f for f in csv_files if Path(f).exists()]
        if not existing_csv_files:
            print("Warning: No generated CSV files found to upload.")
            return

        upload_cmd = [
            sys.executable,
            str(upload_script),
        ] + existing_csv_files + [
            "--type", "DETECTION"
        ]
        print(f"\nUploading generated datasets to VIS and MLP...")
        print(f"Command: {' '.join(upload_cmd)}")
        try:
            subprocess.run(upload_cmd, check=True)
            print(f"Successfully uploaded version {next_version} dataset to MLP.")
        except subprocess.CalledProcessError as e:
            print(f"Error: upload_datasets.py failed with exit code {e.returncode}")
            sys.exit(e.returncode)
    else:
        print(f"\nSkipped uploading datasets (specified --skip-upload).")


if __name__ == "__main__":
    main()
