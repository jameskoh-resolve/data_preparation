#!/usr/bin/env python
"""
Convert curated_issues CSV to validation_data_bbox_add.json for bbox_adder.html GUI.

This prepares data from the curate_dataset_v3 output for manual bbox editing.
Only images with recall issues (missing detections) are flagged for bbox addition.

Usage:
    python prepare_bbox_data.py [--input curated_issues.csv]
"""

import argparse
import csv
import json
from pathlib import Path


def parse_aligned_col(val, col="boxes"):
    """Parse comma-separated column value (special handling for boxes)."""
    if not val or val in ("", "nan"):
        return []
    if col == "boxes":
        results = []
        parts = val.strip().split("],[")
        for i, part in enumerate(parts):
            part = part.strip()
            if i == 0:
                part = part.lstrip("[")
            if i == len(parts) - 1:
                part = part.rstrip("]")
            results.append([int(x) for x in part.split(",")])
        return results
    else:
        return [x.strip() for x in val.split(",")]


def csv_to_json(input_csv, output_json):
    """Convert curated_issues.csv to validation_data_bbox_add.json."""
    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    data = []
    for row_idx, row in enumerate(rows):
        concepts = parse_aligned_col(row.get("concepts", ""), "concepts")
        boxes = parse_aligned_col(row.get("boxes", ""), "boxes")
        issue_types = parse_aligned_col(row.get("issue_types", ""), "issue_types")

        # Determine which concepts are recall issues (model missed them - no box predicted)
        missing_concepts = [
            c for c, it in zip(concepts, issue_types)
            if it.strip() == "recall"
        ]
        missing_concepts = list(dict.fromkeys(missing_concepts))

        filter_reasons = ["recall"] if missing_concepts else []

        # Strip placeholder zero-boxes so they are never rendered on the canvas
        filtered = [
            (c, b, it) for c, b, it in zip(concepts, boxes, issue_types)
            if b != [0, 0, 0, 0]
        ]
        concepts_out = [c for c, b, it in filtered]
        boxes_out = [b for c, b, it in filtered]

        entry = {
            "index": row_idx,
            "im_name": row.get("im_name", f"image_{row_idx}"),
            "im_url": row.get("im_url", ""),
            "concepts": concepts_out,
            "boxes": boxes_out,
            "filter_reasons": filter_reasons,
            "missing_concepts": missing_concepts,
        }
        data.append(entry)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    total_recall = sum(1 for d in data if d["filter_reasons"])
    print(f"Converted {len(data)} rows from {input_csv} -> {output_json}")
    print(f"  {total_recall} images have recall issues (missing bboxes to add)")


def main():
    parser = argparse.ArgumentParser(description="Prepare bbox data for bbox_adder.html GUI.")
    parser.add_argument(
        "--input",
        default=None,
        help="Path to curated_issues CSV (default: auto-detect in parent curation dir)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    output_json = base_dir / "validation_data_bbox_add.json"

    if args.input:
        input_csv = Path(args.input)
    else:
        # Look for curated_issues CSV in the curation directory (one level up)
        # Prefer precision-fixed > final > raw (most processed first)
        curation_dir = base_dir.parent
        candidates = [
            curation_dir / "curated_issues_precision_fixed.csv",
            curation_dir / "curated_issues_final.csv",
            curation_dir / "curated_issues.csv",
        ]
        input_csv = None
        for c in candidates:
            if c.exists():
                input_csv = c
                break
        if input_csv is None:
            print(f"ERROR: No curated_issues CSV found in {curation_dir}")
            print(f"Looked for: {[c.name for c in candidates]}")
            print(f"Run curate_dataset_v3.py first, or specify --input path.")
            return

    if not input_csv.exists():
        print(f"ERROR: {input_csv} not found.")
        return

    csv_to_json(input_csv, output_json)
    html_dir = base_dir.parent / "html"
    print(f"\nNext: Open {html_dir / 'bbox_adder.html'} in a browser")
    print(f"  (serve with: python -m http.server 8000 --directory {base_dir.parent})")


if __name__ == "__main__":
    main()
