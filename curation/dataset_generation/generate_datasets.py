#!/usr/bin/env python
"""
Combined dataset curation, cleaning, distribution generation, and splitting pipeline.
"""

from __future__ import annotations

import csv
import os
import re
import sys
from collections import defaultdict
import typer
from loguru import logger
from path import Path

# Setup paths to import split_datasets
script_dir = Path(os.path.abspath(__file__)).parent
project_root = script_dir.parent.parent

if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from split_datasets import split_dataset

app = typer.Typer(pretty_exceptions_show_locals=False)

CLASSES = [
    "shoe", "top", "product_item", "bottom", "outerwear", "bag", "dress", "eyewear",
    "ethnic_wear", "headwear", "bracelet", "watch", "skirt", "earring", "jewelry_product",
    "scarf", "necklace", "belt", "gloves", "tie", "underwear_bottom", "swimwear",
    "underwear_top", "ring", "onepiece", "hair_accessories", "head_jewelry", "anklet"
]

DATASET_COLS = ["im_url", "im_name", "im_description", "labels", "boxes", "concepts"]
ALIGNED_COLS = ["concepts", "boxes", "optional", "box_statuses", "box_sources", "box_reasons", "issue_types"]


def clip_box_coordinates(boxes_str: str) -> str:
    """Clips box coordinates to be non-negative.
    
    Args:
        boxes_str: String containing box coordinates, e.g. "[ymin,xmin,ymax,xmax],[ymin2,...]"
        
    Returns:
        String of clipped box coordinates.
    """
    if not boxes_str:
        return ""
    box_patterns = re.findall(r"\[[^\]]+\]", boxes_str)
    clipped_boxes = []
    for box in box_patterns:
        coords = [int(x) for x in re.findall(r"-?\d+", box)]
        clipped_coords = [max(0, c) for c in coords]
        if len(clipped_coords) == 4:
            clipped_boxes.append(f"[{clipped_coords[0]},{clipped_coords[1]},{clipped_coords[2]},{clipped_coords[3]}]")
    return ",".join(clipped_boxes)


def parse_box_reasons(val: str) -> list[str]:
    """Robust hybrid parser for box_reasons.
    Handles standard quoting as well as malformed strings missing leading quotes.
    """
    if not val:
        return []
    val = val.strip()
    if val.startswith('['):
        val = val[1:]
    if val.endswith(']'):
        val = val[:-1]
    if not val:
        return []

    # Find standard double-quoted matches or the manual_bbox_added token
    pattern = r'"[^"]*"|manual_bbox_added'
    matches = list(re.finditer(pattern, val))

    elements = []
    last_idx = 0
    for m in matches:
        start, end = m.span()
        unquoted = val[last_idx:start].strip(' ,""')
        if unquoted:
            elements.append(unquoted)
        elements.append(m.group())
        last_idx = end
    unquoted_end = val[last_idx:].strip(' ,""')
    if unquoted_end:
        elements.append(unquoted_end)

    cleaned = []
    for el in elements:
        el = el.strip()
        if el.startswith('"') and el.endswith('"'):
            el = el[1:-1]
        if el:
            cleaned.append(el)
    return cleaned


def serialize_box_reasons(reasons: list[str]) -> str:
    """Serialize a list of reasons back to the dataset bracket format."""
    parts = []
    for r in reasons:
        if r == "manual_bbox_added":
            parts.append(r)
        else:
            escaped = r.replace('"', '""')
            parts.append(f'"{escaped}"')
    return "[" + ",".join(parts) + "]"


def split_col(row: dict[str, str], col: str) -> list[str]:
    """Split an aligned column into parts."""
    val = row.get(col, "")
    if col == "boxes":
        return re.findall(r"\[[^\]]+\]", val)
    elif col == "box_reasons":
        return parse_box_reasons(val)
    else:
        return val.split(",") if val else []


def compact_aligned_columns(row: dict[str, str]) -> tuple[dict[str, str], int]:
    """Remove entries from aligned columns where the concept is empty.

    Corrections that remove entries can leave residual commas in comma-separated
    columns (e.g. concepts becomes ",,,eyewear,top").  The boxes regex parser
    skips empty bracket slots but split(",") on concepts preserves them, causing
    index misalignment.  This function compacts all aligned columns by keeping
    only the indices where the concept is non-empty.

    Returns:
        Tuple of (compacted_row, num_empty_entries_removed).
    """
    concepts = split_col(row, "concepts")
    non_empty_indices = [i for i, c in enumerate(concepts) if c.strip()]

    if len(non_empty_indices) == len(concepts):
        return row, 0

    removed = len(concepts) - len(non_empty_indices)
    new_row = dict(row)
    for col in ALIGNED_COLS:
        if col not in row:
            continue
        parts = split_col(row, col)
        kept = [parts[i] for i in non_empty_indices if i < len(parts)]
        if col == "box_reasons":
            new_row[col] = serialize_box_reasons(kept)
        else:
            new_row[col] = ",".join(kept)

    return new_row, removed


def clean_row(row: dict[str, str]) -> tuple[dict[str, str] | None, str, int]:
    """Cleans a single CSV row using index-based filtering.
    
    Returns:
        Tuple of (new_row, status, dupes_removed) where status is 'kept', 'empty', or 'single_other'.
    """
    # Compact first to fix any misaligned empty concept slots.
    row, _ = compact_aligned_columns(row)

    boxes = split_col(row, "boxes")
    concepts = split_col(row, "concepts")

    seen = set()
    remaining_indices = []
    dupes_removed = 0

    for i, b in enumerate(boxes):
        b_clean = b.replace(" ", "")
        
        if b_clean == "[0,0,0,0]":
            continue
            
        c = concepts[i] if i < len(concepts) else ""
        key = (b_clean, c)
        if key in seen:
            dupes_removed += 1
            continue
            
        seen.add(key)
        remaining_indices.append(i)

    # Filtering check 1: Remove images with 0 remaining boxes
    if not remaining_indices:
        return None, "empty", dupes_removed

    # Filtering check 2: Remove images with exactly 1 remaining box labelled "other"
    if len(remaining_indices) == 1:
        idx = remaining_indices[0]
        concept = concepts[idx] if idx < len(concepts) else ""
        if concept == "other":
            return None, "single_other", dupes_removed

    # Reconstruct the cleaned row by extracting kept indices from all aligned columns
    new_row = dict(row)
    for col in ALIGNED_COLS:
        if col not in row:
            continue
        parts = split_col(row, col)
        cleaned_parts = []
        for idx in remaining_indices:
            if idx < len(parts):
                cleaned_parts.append(parts[idx])
            else:
                cleaned_parts.append("")

        if col == "box_reasons":
            new_row[col] = serialize_box_reasons(cleaned_parts)
        else:
            new_row[col] = ",".join(cleaned_parts)

    return new_row, "kept", dupes_removed


def verify_dataset(file_path: Path) -> None:
    """Perform self-verification on the output file to ensure curation rules were correctly applied."""
    logger.info(f"Running self-verification on {file_path.name}...")
    verification_failed = False
    
    with file_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for idx, row in enumerate(rows):
        boxes_str = row.get("boxes", "")
        concepts_str = row.get("concepts", "")
        
        boxes = re.findall(r"\[[^\]]+\]", boxes_str)
        concepts = [c.strip() for c in concepts_str.split(",") if c.strip()]
        
        # Check coordinates are positive/non-negative
        for b in boxes:
            coords = [int(x) for x in re.findall(r"-?\d+", b)]
            if any(c < 0 for c in coords):
                logger.error(f"Row {idx} ({row.get('im_name', '')}) contains negative coordinates: {b}")
                verification_failed = True
            if coords == [0, 0, 0, 0]:
                logger.error(f"Row {idx} ({row.get('im_name', '')}) contains [0,0,0,0] box!")
                verification_failed = True
                
        # Check duplicate boxes (concept + coordinates)
        seen = set()
        for i, b in enumerate(boxes):
            c = concepts[i] if i < len(concepts) else ""
            key = (b.replace(" ", ""), c)
            if key in seen:
                logger.error(f"Row {idx} ({row.get('im_name', '')}) contains duplicate box: {key}")
                verification_failed = True
            seen.add(key)
            
        # Check for empty boxes
        if len(boxes) == 0:
            logger.error(f"Row {idx} ({row.get('im_name', '')}) has 0 boxes!")
            verification_failed = True
            
        # Check for single 'other' box
        if len(boxes) == 1 and len(concepts) == 1 and concepts[0] == "other":
            logger.error(f"Row {idx} ({row.get('im_name', '')}) has exactly 1 'other' box!")
            verification_failed = True

        # Check for empty concept strings (residual commas from corrections)
        raw_concepts = [c.strip() for c in concepts_str.split(",")]
        empty_concepts = [i for i, c in enumerate(raw_concepts) if not c]
        if empty_concepts:
            logger.error(f"Row {idx} ({row.get('im_name', '')}) has empty concept(s) at indices {empty_concepts}")
            verification_failed = True

        # Check concept-box count alignment
        if len(boxes) != len(concepts):
            logger.error(
                f"Row {idx} ({row.get('im_name', '')}) has mismatched counts: "
                f"{len(concepts)} concepts vs {len(boxes)} boxes"
            )
            verification_failed = True

    if verification_failed:
        raise ValueError("Dataset verification failed! Please check logs for details.")
    logger.info("Dataset verification passed successfully!")


@app.command()
def main(
    input_csv_path: str = typer.Argument(..., help="Path to the input curated CSV (e.g. curated_issues_final.csv)"),
    output_folder: str = typer.Argument(..., help="Path to the output folder"),
    dataset_name: str = typer.Argument(..., help="Name of the dataset (e.g. veromoda_v1)"),
) -> None:
    """Generate curated dataset, distribution stats, and train/val/test splits."""
    input_path = Path(input_csv_path)
    output_dir = Path(output_folder)
    
    if not input_path.exists():
        logger.error(f"Input file does not exist: {input_path}")
        sys.exit(1)
        
    output_dir.makedirs_p()

    prefix = f"{dataset_name}_dataset"
    dataset_path = output_dir / f"{prefix}.csv"
    distribution_path = output_dir / f"{prefix}_data_distribution.csv"

    logger.info(f"Reading input curated file: {input_path}")
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info(f"Loaded {len(rows)} rows. Cleaning and processing...")

    cleaned_rows = []
    removed_zeros = 0
    removed_dupes = 0
    removed_empty = 0
    removed_single_other = 0
    compacted_empty = 0

    for idx, row in enumerate(rows):
        # Compact empty concept slots left by corrections before counting zeros.
        row, n_compacted = compact_aligned_columns(row)
        compacted_empty += n_compacted

        # Count zeros removed in this row
        boxes_before = split_col(row, "boxes")
        zeros_count = sum(1 for b in boxes_before if b.replace(" ", "") == "[0,0,0,0]")
        removed_zeros += zeros_count

        new_row, status, dupes_removed = clean_row(row)
        removed_dupes += dupes_removed

        if status == "kept":
            assert new_row is not None
            cleaned_rows.append(new_row)
        elif status == "empty":
            removed_empty += 1
        elif status == "single_other":
            removed_single_other += 1

    logger.info("Processing Statistics:")
    logger.info(f"  Total input rows: {len(rows)}")
    logger.info(f"  Total empty concept entries compacted: {compacted_empty}")
    logger.info(f"  Total '[0,0,0,0]' boxes removed: {removed_zeros}")
    logger.info(f"  Total duplicate boxes removed: {removed_dupes}")
    logger.info(f"  Rows removed because they ended up empty: {removed_empty}")
    logger.info(f"  Rows removed because they had only 1 'other' box: {removed_single_other}")
    logger.info(f"  Total rows kept: {len(cleaned_rows)}")

    # Format for output dataset
    logger.info(f"Writing curated full dataset to: {dataset_path}")
    with dataset_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DATASET_COLS)
        writer.writeheader()
        for row in cleaned_rows:
            out_row = {col: row.get(col, "") for col in DATASET_COLS}
            out_row["im_description"] = ""  # Empty by default
            out_row["boxes"] = clip_box_coordinates(row.get("boxes", ""))
            writer.writerow(out_row)

    # Perform self-verification
    verify_dataset(dataset_path)

    # Compute and write distribution statistics
    logger.info(f"Generating data distribution stats to: {distribution_path}")
    box_count = defaultdict(int)
    image_set = defaultdict(set)

    for idx, row in enumerate(cleaned_rows):
        concepts = [c.strip() for c in row.get("concepts", "").split(",") if c.strip()]
        for concept in concepts:
            box_count[concept] += 1
            image_set[concept].add(idx)

    extra_classes = set(box_count.keys()) - set(CLASSES)
    if extra_classes:
        logger.warning(f"Detected classes in dataset not in standard 28 fashion classes list: {extra_classes}")

    with distribution_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["concept", "# boxes", "# images"])
        for cls in CLASSES:
            writer.writerow([cls, box_count.get(cls, 0), len(image_set.get(cls, set()))])

    total_boxes = sum(box_count.get(cls, 0) for cls in CLASSES)
    logger.info(f"Distribution: {len(CLASSES)} classes, {total_boxes} total boxes")

    # Split dataset
    logger.info(f"Calling split_dataset on {dataset_path}...")
    split_dataset(dataset_path, prefix, output_dir)
    logger.info("Curation and splitting pipeline completed successfully!")


if __name__ == "__main__":
    app()
