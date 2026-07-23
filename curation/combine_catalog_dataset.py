#!/usr/bin/env python
"""Combine per-catalog human_worn_sample.csv files into one flat CSV for auto_annotate.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

KEEP_COLS = ["catalog_name", "product_id", "title", "brand", "colour", "category", "im_url", "im_id"]


@app.command()
def main(
    root: str = typer.Argument("curated_datasets/combined_catalogs/catalog_dataset_part_1"),
    output: str = typer.Option("curated_datasets/combined_catalogs/catalog_dataset_part_1/combined.csv"),
):
    root_path = Path(root)
    frames = []
    for csv_path in sorted(root_path.glob("*/human_worn_sample.csv")):
        catalog = csv_path.parent.name
        df = pd.read_csv(csv_path)
        df = df[KEEP_COLS]
        frames.append(df)
        print(f"{catalog}: {len(df)} rows")

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset="im_id")
    print(f"Total: {before} rows ({before - len(combined)} duplicate im_id dropped) -> {len(combined)} rows")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    app()
