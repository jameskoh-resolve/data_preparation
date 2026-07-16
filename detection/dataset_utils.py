"""Dataset parsing and IO helpers for detection evaluation."""

import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
from loguru import logger

from utils.vis_image import get_image_content


def url_to_slug(url: str) -> str:
    """Build a stable image slug from URL path stem."""
    path = urlparse(url).path
    stem = Path(path).stem[:80]
    return stem if stem else "image"


def download_image(url: str, cache_dir: Path):
    """Download and cache image, returning decoded BGR image and cache path.
    
    Uses get_image_content for robust handling of ASOS headers, VIS caching,
    and proper error handling.
    """
    slug = url_to_slug(url)
    cache_path = cache_dir / f"{slug}.jpg"
    
    if cache_path.exists():
        try:
            image_bytes = cache_path.read_bytes()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                return image, cache_path
        except Exception:
            pass

    # Fetch image bytes with smart handling (ASOS headers, VIS, local files, caching)
    image_bytes = get_image_content(url, timeout=20)
    if image_bytes is None:
        print(f"  WARNING: could not download {url}", file=sys.stderr)
        return None, None
    
    # Decode from bytes to BGR
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(image_bytes)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"  WARNING: could not decode image from {url}", file=sys.stderr)
            return None, None
    except Exception as exc:
        print(f"  WARNING: could not process image from {url}: {exc}", file=sys.stderr)
        return None, None
    
    return image, cache_path


def parse_outfits(raw: str) -> list:
    """Parse serialized outfits list from CSV cell."""
    try:
        return ast.literal_eval(raw)
    except Exception:
        return []


def _parse_flat_boxes(raw_boxes: str) -> List[List[float]]:
    """Parse a flat CSV boxes field into list of [x1, y1, x2, y2] boxes."""
    if pd.isna(raw_boxes):
        return []

    raw = str(raw_boxes).strip()
    if raw == "":
        return []

    matches = re.findall(r"\[[^\]]+\]", raw)
    if matches:
        out = []
        for match in matches:
            box = parse_box(match)
            if box is not None:
                out.append(box)
            else:
                logger.warning(f"Failed to parse box value in flat GT: {match}")
        return out

    single = parse_box(raw)
    return [single] if single is not None else []


def _parse_flat_concepts(raw_concepts: str) -> List[str]:
    """Parse a flat CSV concepts field into list of class names."""
    if pd.isna(raw_concepts):
        return []

    raw = str(raw_concepts).strip()
    if raw == "":
        return []

    return [part.strip() for part in raw.split(",") if part.strip()]


def build_detection_gt_from_flat_csv(
    flat_gt_file: str,
    product_key_col: str = "im_name",
    image_id_col: str = "im_id",
    image_url_col: str = "im_url",
    concepts_col: str = "concepts",
    boxes_col: str = "boxes",
) -> Dict[str, Dict[str, List[dict]]]:
    """Build canonical detection GT grouping from a flat CSV.

    Returns:
        product_key -> im_url -> [{"box": [x1, y1, x2, y2], "box_concept": concept}, ...]
    """
    flat_gt_path = Path(flat_gt_file)
    if not flat_gt_path.exists():
        raise FileNotFoundError(f"Flat GT file not found: {flat_gt_file}")

    df = pd.read_csv(flat_gt_path, dtype=str)

    required_cols = [image_url_col, concepts_col, boxes_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Flat GT file is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    grouped: Dict[str, Dict[str, List[dict]]] = {}
    fallback_counter = 0

    for row_idx, row in df.iterrows():
        im_url = str(row.get(image_url_col, "") or "").strip()
        if im_url == "" or im_url == "nan":
            continue

        product_key = str(row.get(product_key_col, "") or "").strip()
        if product_key == "" or product_key == "nan":
            product_key = str(row.get(image_id_col, "") or "").strip()
        if product_key == "" or product_key == "nan":
            product_key = f"unknown_{fallback_counter}"
            fallback_counter += 1

        concepts = _parse_flat_concepts(row.get(concepts_col, ""))
        boxes = _parse_flat_boxes(row.get(boxes_col, ""))

        if not concepts or not boxes:
            continue

        pair_count = min(len(concepts), len(boxes))
        if len(concepts) != len(boxes):
            logger.warning(
                f"Flat GT row {row_idx} has concept/box length mismatch "
                f"({len(concepts)} vs {len(boxes)}). Truncating to {pair_count}."
            )

        seen = set()
        bucket = grouped.setdefault(product_key, {}).setdefault(im_url, [])
        existing = {(tuple(item["box"]), item["box_concept"]) for item in bucket}
        for i in range(pair_count):
            concept = concepts[i]
            box = boxes[i]
            dedup_key = (tuple(box), concept)
            if dedup_key in existing or dedup_key in seen:
                continue
            seen.add(dedup_key)
            bucket.append({"box": box, "box_concept": concept})

    return grouped


def parse_box(raw_box: str):
    """Parse comma-separated box into [x1, y1, x2, y2] floats."""
    try:
        parts = [float(v) for v in str(raw_box).split(",")]
    except Exception:
        return None
    if len(parts) != 4:
        return None
    return parts


def flatten_dataset(
    df,
    gt_concept_legacy_map: Dict[str, str],
    product_filter=None,
    gt_format: str = "nested",
    flat_gt_file: Optional[Path] = None,
):
    """Load GT from nested or flat source and return product->image->items mapping."""
    if gt_format == "flat":
        if flat_gt_file is None:
            raise ValueError("flat_gt_file is required when gt_format='flat'")

        grouped = build_detection_gt_from_flat_csv(str(flat_gt_file))
    else:
        grouped = {}
        for _, row in df.iterrows():
            product_id = str(row["product_id"])
            outfits = parse_outfits(str(row.get("search_gt_outfits", "")))
            for item in outfits:
                im_url = item.get("im_url", "")
                box = parse_box(item.get("box", ""))
                if not im_url or box is None:
                    continue
                raw_concept = str(item.get("box_concept", ""))
                box_concept = gt_concept_legacy_map.get(raw_concept, raw_concept)
                grouped.setdefault(product_id, {}).setdefault(im_url, []).append(
                    {"box": box, "box_concept": box_concept}
                )

    filtered_grouped = {}
    for product_id in grouped:
        if product_filter and product_id not in product_filter:
            continue
        for im_url in grouped[product_id]:
            items = []
            for item in grouped[product_id][im_url]:
                box = item.get("box") if isinstance(item, dict) else None
                if box is None:
                    box = parse_box(item.get("box", ""))
                if box is None:
                    continue
                raw_concept = str(item.get("box_concept", ""))
                box_concept = gt_concept_legacy_map.get(raw_concept, raw_concept)
                items.append({"box": box, "box_concept": box_concept})

            if not items:
                continue
            filtered_grouped.setdefault(product_id, {})[im_url] = items

    return filtered_grouped


def build_image_list_from_catalog(catalog_df: pd.DataFrame, product_filter=None):
    """Build product->{im_url: []} structure from catalog (no GT items yet)."""
    grouped = {}
    for _, row in catalog_df.iterrows():
        product_id = str(row["product_id"])
        if product_filter and product_id not in product_filter:
            continue
        main_url = str(row.get("main_image_url", "")).strip()
        if main_url:
            grouped.setdefault(product_id, {})[main_url] = []
        additional = str(row.get("additional_image_url", "")).strip()
        if additional and additional != "nan":
            for url in additional.split(","):
                url = url.strip()
                if url:
                    grouped.setdefault(product_id, {})[url] = []
    return grouped


def save_gt_to_csv(gt_rows: list, output_csv: Path):
    """Save GT rows back to query_annotated.csv format for future runs without LLM."""
    by_product: Dict[str, list] = {}
    for row in gt_rows:
        by_product.setdefault(row["product_id"], []).append(
            {
                "im_url": row["im_url"],
                "box_concept": row["gt_box_concept"],
                "box": row["box"],
            }
        )

    records = []
    for product_id, items in sorted(by_product.items()):
        records.append(
            {
                "product_id": product_id,
                "search_gt_outfits": json.dumps(items),
            }
        )

    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)


def extract_box_array(record: dict):
    """Extract box array from either box_list or CSV box string."""
    box = record.get("box_list")
    if box is not None:
        return [float(v) for v in box]
    parsed = parse_box(record.get("box", ""))
    if parsed is None:
        return None
    return [float(v) for v in parsed]
