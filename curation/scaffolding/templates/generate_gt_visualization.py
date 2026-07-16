#!/usr/bin/env python
"""
Generate a GT bounding box visualization for the dataset.
Reads curated_issues_final.csv and produces:
  - gt_visualization_curated_issues_final.json (lightweight data for the viewer)
"""

import csv
import json
from pathlib import Path


def parse_boxes(boxes_str):
    """Parse '[x1,y1,x2,y2],[x3,y3,x4,y4]' into list of [x1,y1,x2,y2] lists."""
    if not boxes_str or boxes_str in ("", "nan"):
        return []
    results = []
    parts = boxes_str.replace("],[", "]|[").split("|")
    for part in parts:
        part = part.replace("[", "").replace("]", "").strip()
        if not part:
            continue
        tokens = [x.strip() for x in part.split(",") if x.strip()]
        if not tokens:
            continue
        try:
            coords = [int(x) for x in tokens]
            if len(coords) == 4:
                results.append(coords)
        except ValueError:
            continue
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate GT visualization JSON from a curation CSV.")
    parser.add_argument(
        "csv_input",
        nargs="?",
        default=None,
        help="Path to the input curation CSV file. Defaults to curated_issues_final.csv in the parent directory."
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.csv_input:
        csv_path = Path(args.csv_input).resolve()
    else:
        csv_path = base_dir.parent / "curated_issues_final.csv"

    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return

    json_output = base_dir / "gt_visualization_curated_issues_final.json"

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    data = []
    for idx, row in enumerate(rows):
        concepts = [c.strip() for c in row.get("concepts", "").split(",") if c.strip()]
        boxes = parse_boxes(row.get("boxes", ""))
        im_url = row.get("im_url", "")
        im_name = row.get("im_name", "")
        im_id = row.get("im_id", "")

        # Build annotations list (concept + box pairs)
        annotations = []
        for i, concept in enumerate(concepts):
            box = boxes[i] if i < len(boxes) else None
            if box and box != [0, 0, 0, 0]:
                annotations.append({"concept": concept, "box": box})

        data.append({
            "index": idx,
            "im_id": im_id,
            "im_url": im_url,
            "im_name": im_name,
            "annotations": annotations,
        })

    with json_output.open("w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"Generated {json_output.name}: {len(data)} images, {sum(len(d['annotations']) for d in data)} total GT boxes")


if __name__ == "__main__":
    main()
