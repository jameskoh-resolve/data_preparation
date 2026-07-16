#!/usr/bin/env python
"""
Unified GT bounding box correction script.

Applies add/remove/replace/replace_multi actions to curated_issues_final.csv.

Usage:
    python apply_gt_corrections.py corrections.json

JSON format:
[
    {
        "row_index": 259,
        "action": "add",
        "concept": "bracelet",
        "box": [x1, y1, x2, y2]
    },
    {
        "row_index": 100,
        "action": "remove",
        "concept_index": 2
    },
    {
        "row_index": 200,
        "action": "replace",
        "concept_index": 1,
        "new_concept": "bottom",
        "new_box": [x1, y1, x2, y2]
    },
    {
        "row_index": 300,
        "action": "replace_multi",
        "concept_index": 0,
        "new_boxes": [{"concept": "top", "box": [x1, y1, x2, y2]}, ...]
    }
]

Actions:
  - "add": Append a new bounding box entry to the row.
  - "remove": Remove the entry at concept_index from all aligned columns.
  - "replace": Replace the concept/box at concept_index.
  - "replace_multi": Remove entry at concept_index, then append multiple new entries.
"""

import csv
import json
import re
import sys
from pathlib import Path

ALIGNED_COLS = ["concepts", "boxes", "optional", "box_statuses", "box_sources", "box_reasons", "issue_types"]


def split_col(row, col):
    """Split an aligned column into parts."""
    val = row.get(col, "")
    if col == "boxes":
        return re.findall(r"\[[^\]]+\]", val)
    elif col == "box_reasons":
        return re.findall(r'"[^"]*"|manual_bbox_added', val.strip("[]"))
    else:
        return val.split(",") if val else []


def join_col(parts, col):
    """Join parts back into CSV string for a column."""
    if col == "box_reasons":
        return "[" + ",".join(parts) + "]"
    return ",".join(parts)


def remove_at_index(row, concept_idx):
    """Remove entry at concept_idx from all aligned columns."""
    for col in ALIGNED_COLS:
        parts = split_col(row, col)
        parts = [p for i, p in enumerate(parts) if i != concept_idx]
        row[col] = join_col(parts, col)


def append_entry(row, concept, box):
    """Append a new concept+box entry to the row."""
    row["concepts"] = row["concepts"] + "," + concept if row["concepts"] else concept
    box_str = f"[{box[0]},{box[1]},{box[2]},{box[3]}]"
    row["boxes"] = row["boxes"] + "," + box_str if row["boxes"] else box_str
    row["optional"] = row["optional"] + ",true" if row["optional"] else "true"
    row["box_statuses"] = row["box_statuses"] + ",human_drawn" if row["box_statuses"] else "human_drawn"
    row["box_sources"] = row["box_sources"] + ",human" if row["box_sources"] else "human"
    row["box_reasons"] = row["box_reasons"] + ",manual_bbox_added" if row["box_reasons"] else "manual_bbox_added"
    row["issue_types"] = row["issue_types"] + ",no_issue" if row["issue_types"] else "no_issue"


def main():
    base_dir = Path(__file__).resolve().parent
    corrections_file = sys.argv[1] if len(sys.argv) > 1 else "gt_box_corrections.json"
    csv_path = base_dir.parent / "curated_issues_final.csv"
    output_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else csv_path

    if not csv_path.exists():
        import shutil
        source_path = base_dir.parent / "curated_issues_precision_fixed.csv"
        print(f"Initializing {csv_path.name} from {source_path.name}")
        shutil.copy(source_path, csv_path)

    with open(base_dir / corrections_file, "r", encoding="utf-8") as f:
        corrections = json.load(f)

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for corr in corrections:
        row_idx = corr["row_index"]
        action = corr["action"]
        row = rows[row_idx]

        if action == "add":
            concept = corr.get("concept", "")
            box = corr["box"]
            append_entry(row, concept, box)
            print(f"Row {row_idx}: added {concept} with box {box}")

        elif action == "remove":
            concept_idx = corr["concept_index"]
            remove_at_index(row, concept_idx)
            print(f"Row {row_idx}: removed entry at index {concept_idx}")

        elif action == "replace":
            concept_idx = corr["concept_index"]
            new_concept = corr.get("new_concept")
            new_box = corr.get("new_box")

            if new_concept:
                concepts = row["concepts"].split(",")
                concepts[concept_idx] = new_concept
                row["concepts"] = ",".join(concepts)

            if new_box:
                box_parts = re.findall(r"\[[^\]]+\]", row["boxes"])
                box_parts[concept_idx] = f"[{new_box[0]},{new_box[1]},{new_box[2]},{new_box[3]}]"
                row["boxes"] = ",".join(box_parts)

            # Update metadata
            issues = row["issue_types"].split(",")
            while len(issues) <= concept_idx:
                issues.append("")
            issues[concept_idx] = "no_issue"
            row["issue_types"] = ",".join(issues)

            sources = row.get("box_sources", "").split(",")
            while len(sources) <= concept_idx:
                sources.append("")
            sources[concept_idx] = "human"
            row["box_sources"] = ",".join(sources)

            statuses = row.get("box_statuses", "").split(",")
            while len(statuses) <= concept_idx:
                statuses.append("")
            statuses[concept_idx] = "human_drawn"
            row["box_statuses"] = ",".join(statuses)

            print(f"Row {row_idx}: replaced index {concept_idx} -> {new_concept} with box {new_box}")

        elif action == "replace_multi":
            concept_idx = corr["concept_index"]
            new_boxes = corr.get("new_boxes", [])

            # Remove original slot
            remove_at_index(row, concept_idx)

            if not new_boxes:
                print(f"Row {row_idx}: removed entry (no replacements)")
            else:
                # Append new entries
                for entry in new_boxes:
                    append_entry(row, entry["concept"], entry["box"])
                print(f"Row {row_idx}: replaced with {len(new_boxes)} new boxes: {[b['concept'] for b in new_boxes]}")

    # Clean up [0,0,0,0] boxes for any row that had corrections
    modified_rows = set(corr["row_index"] for corr in corrections)
    for row_idx in modified_rows:
        row = rows[row_idx]
        boxes = split_col(row, "boxes")
        indices_to_remove = [i for i, b in enumerate(boxes) if b.replace(" ", "") == "[0,0,0,0]"]
        for idx in sorted(indices_to_remove, reverse=True):
            remove_at_index(row, idx)
            print(f"Row {row_idx}: automatically removed unresolved [0,0,0,0] box at index {idx}")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nApplied {len(corrections)} corrections to {output_path}")


if __name__ == "__main__":
    main()
