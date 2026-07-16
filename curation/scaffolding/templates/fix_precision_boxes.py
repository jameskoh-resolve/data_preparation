#!/usr/bin/env python
"""
Auto-fix precision bounding boxes in the dataset.

Rules:
- any concept with precision/fp_high_conf_incorrect -> REMOVE
"""

import csv
from pathlib import Path


PRECISION_ISSUES = {"precision", "fp_high_conf_incorrect"}

# Columns that are aligned per-concept (comma-separated, same length)
ALIGNED_COLUMNS = ["concepts", "boxes", "optional", "box_statuses", "box_sources", "box_reasons", "issue_types"]


def parse_boxes(boxes_str):
    """Parse '[x1,y1,x2,y2],[x3,y3,x4,y4]' into list of box strings."""
    if not boxes_str or boxes_str in ("", "nan"):
        return []
    results = []
    parts = boxes_str.strip().split("],[")
    for i, part in enumerate(parts):
        part = part.strip()
        if i == 0:
            part = part.lstrip("[")
        if i == len(parts) - 1:
            part = part.rstrip("]")
        results.append(f"[{part}]")
    return results


def parse_aligned(row, col):
    """Parse a comma-separated aligned column, handling box format specially."""
    val = row.get(col, "")
    if not val or val == "nan":
        return []
    if col == "boxes":
        return parse_boxes(val)
    return [x.strip() for x in val.split(",")]


def serialize_aligned(values, col):
    """Serialize aligned column values back to CSV format."""
    return ",".join(values)


def fix_row(row, row_idx, log):
    """Fix precision issues in a single row. Returns modified row."""
    concepts = parse_aligned(row, "concepts")
    issue_types = parse_aligned(row, "issue_types")

    if not concepts or not issue_types:
        return row

    # Build per-column aligned lists
    aligned = {}
    for col in ALIGNED_COLUMNS:
        aligned[col] = parse_aligned(row, col)

    # Pad shorter lists to match concepts length
    n = len(concepts)
    for col in ALIGNED_COLUMNS:
        while len(aligned[col]) < n:
            aligned[col].append("")

    # Indices to remove
    indices_to_remove = set()

    for i in range(n):
        issue = aligned["issue_types"][i]
        concept = aligned["concepts"][i]

        if issue not in PRECISION_ISSUES:
            continue

        indices_to_remove.add(i)
        log.append(f"  Row {row_idx}: REMOVED {concept} ({issue})")

    # Remove marked indices (preserve order)
    if indices_to_remove:
        for col in ALIGNED_COLUMNS:
            aligned[col] = [v for idx, v in enumerate(aligned[col]) if idx not in indices_to_remove]

    # Write back
    new_row = dict(row)
    for col in ALIGNED_COLUMNS:
        new_row[col] = serialize_aligned(aligned[col], col)

    return new_row


def main():
    base_dir = Path(__file__).resolve().parent
    input_path = base_dir.parent / "curated_issues.csv"
    output_path = base_dir.parent / "curated_issues_precision_fixed.csv"

    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    log = []
    fixed_rows = []

    for row_idx, row in enumerate(rows):
        new_row = fix_row(row, row_idx, log)
        fixed_rows.append(new_row)

    removed_count = sum(1 for e in log if "REMOVED" in e)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fixed_rows)

    print(f"Input: {input_path} ({len(rows)} rows)")
    print(f"Output: {output_path} ({len(fixed_rows)} rows)")
    print(f"\nActions:")
    print(f"  Boxes removed: {removed_count}")

    if log:
        print(f"\nFull log ({len(log)} actions):")
        for entry in log:
            print(entry)


if __name__ == "__main__":
    main()
