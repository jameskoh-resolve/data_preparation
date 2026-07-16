#!/usr/bin/env python
"""Category sampler for ASOS v1 dataset.

Reads a filter spec YAML (asos_combined_filter_spec.yaml) that defines which
categories to include — optionally gated by a gender field — then expands each
product into individual image rows (main + additional images), applies a
proportional-capped budget allocation across categories, and writes a pool CSV.

The output pool CSV is intended as input to human_presence_sampling.py so that
the final 2000-image dataset contains only images where a human model is present.

Usage:
    python scripts/curation/category_sampler.py configs/curation/asos_v1_sampling.yaml
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd
import typer
from loguru import logger
from omegaconf import OmegaConf
from path import Path
from tqdm import tqdm

app = typer.Typer(pretty_exceptions_show_locals=False)

REPO_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.isabs() else REPO_ROOT / path


def _parse_list_field(cell: Any) -> List[str]:
    """Parse a stringified Python list / JSON array into a list of strings."""
    if cell is None:
        return []
    if isinstance(cell, list):
        return [str(x).strip() for x in cell]
    text = str(cell).strip()
    if not text:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed]
        except Exception:
            pass
    return [text]


def _category_from_cell(cell: Any) -> str:
    """Return the primary category string from a cell that may be a list."""
    items = _parse_list_field(cell)
    return items[0] if items else ""


def _expand_images(df: pd.DataFrame, image_col: str, additional_col: Optional[str]) -> pd.DataFrame:
    """Expand each product row into one row per image URL (main + additional).

    Returns a DataFrame with all original columns plus an `im_url` column.
    Duplicate URLs within the same product are deduplicated.
    """
    rows: List[Dict[str, Any]] = []
    for record in tqdm(df.to_dict("records"), desc="Expanding images", unit="product"):
        seen: set[str] = set()
        candidates: List[str] = []

        main = str(record.get(image_col, "") or "").strip()
        if main.startswith(("http://", "https://")):
            candidates.append(main)

        if additional_col and additional_col in record:
            for url in _parse_list_field(record.get(additional_col)):
                url = url.strip()
                if url.startswith(("http://", "https://")):
                    candidates.append(url)

        for url in candidates:
            if url not in seen:
                seen.add(url)
                row = dict(record)
                row["im_url"] = url
                rows.append(row)

    return pd.DataFrame(rows)


def _apply_filter_spec(df: pd.DataFrame, filter_spec_path: Path, category_col: str, gender_col: str) -> pd.DataFrame:
    """Apply catalog_filter rules from the filter spec YAML.

    Supports two rule types:
      - category_in (no gender restriction)
      - category_in + gender_equals (women-only gate)

    Returns a deduplicated DataFrame with an added `matched_rule` column.
    """
    spec = OmegaConf.load(str(filter_spec_path))
    rules = spec.catalog_filter.rules

    # Normalise category column: extract first element if it's a list string
    df = df.copy()
    df["_category_norm"] = df[category_col].apply(_category_from_cell).str.strip().str.lower()
    df["_gender_str"] = df[gender_col].astype(str).str.strip() if gender_col in df.columns else ""

    accepted_frames: List[pd.DataFrame] = []
    accepted_ids: set = set()  # im_url dedup across rules

    for rule in rules:
        rule_name = str(rule.name)
        category_in = [str(c).strip().lower() for c in (rule.get("category_in") or [])]
        gender_equals = str(rule.get("gender_equals", "")).strip()

        if not category_in:
            continue

        mask_cat = df["_category_norm"].isin(category_in)

        if gender_equals:
            mask_gender = df["_gender_str"] == gender_equals
            mask = mask_cat & mask_gender
        else:
            mask = mask_cat

        matched = df[mask].copy()
        # Exclude rows already accepted by a previous rule
        if accepted_ids:
            matched = matched[~matched["im_url"].isin(accepted_ids)]

        if matched.empty:
            logger.info("Rule '{}': 0 new rows", rule_name)
            continue

        matched["matched_rule"] = rule_name
        matched["category_clean"] = matched["_category_norm"].str.title()
        accepted_ids.update(matched["im_url"].tolist())
        accepted_frames.append(matched)
        logger.info("Rule '{}': {} image rows matched", rule_name, len(matched))

    if not accepted_frames:
        logger.warning("No rows matched any filter rule — check your filter spec and catalog.")
        return pd.DataFrame()

    result = pd.concat(accepted_frames, ignore_index=True)
    result = result.drop(columns=["_category_norm", "_gender_str"])
    return result


def _allocate_budget(
    category_counts: Dict[str, int],
    total_target: int,
    min_per_category: int,
    max_per_category: int,
) -> Dict[str, int]:
    """Proportional-capped budget allocation using sqrt(pool_size) weights.

    Args:
        category_counts: mapping of category -> available image count.
        total_target: total images to allocate across all categories.
        min_per_category: floor allocation (clamped to actual pool size).
        max_per_category: ceiling allocation per category.

    Returns:
        Mapping of category -> allocated image count.
    """
    categories = list(category_counts.keys())
    pools = {c: category_counts[c] for c in categories}

    # Initial proportional allocation via sqrt weights
    weights = {c: math.sqrt(max(pools[c], 1)) for c in categories}
    total_weight = sum(weights.values())

    alloc: Dict[str, int] = {}
    for c in categories:
        raw = int(round(total_target * weights[c] / total_weight))
        alloc[c] = max(min_per_category, min(max_per_category, raw, pools[c]))

    # Redistribute remainder to under-filled categories (sorted by pool desc)
    for _ in range(10):  # iterate to converge
        current_total = sum(alloc.values())
        remainder = total_target - current_total
        if remainder == 0:
            break
        candidates = sorted(
            [c for c in categories if alloc[c] < min(max_per_category, pools[c])],
            key=lambda c: pools[c],
            reverse=True,
        )
        if not candidates or remainder <= 0:
            break
        give = remainder // len(candidates) or 1
        for c in candidates:
            headroom = min(max_per_category, pools[c]) - alloc[c]
            add = min(give, headroom)
            alloc[c] += add
            remainder -= add
            if remainder <= 0:
                break

    return alloc


def _resolve_per_category_targets(
    raw_targets: Dict[str, Any],
    category_pool_sizes: Dict[str, int],
) -> Dict[str, int]:
    """Resolve case-insensitive per-category targets from config.

    Args:
        raw_targets: config mapping category_name -> requested sample count.
        category_pool_sizes: available image counts by sampled category.

    Returns:
        Mapping of canonical sampled category name -> requested target count.
    """
    if not raw_targets:
        return {}

    cat_lookup = {str(c).strip().lower(): c for c in category_pool_sizes.keys()}
    resolved: Dict[str, int] = {}

    for raw_cat, raw_value in raw_targets.items():
        key = str(raw_cat).strip().lower()
        if key not in cat_lookup:
            logger.warning("per_category_target ignored: '{}' not found in sampled categories", raw_cat)
            continue

        try:
            target = int(raw_value)
        except Exception:
            logger.warning("per_category_target ignored: '{}' has non-integer value {}", raw_cat, raw_value)
            continue

        if target < 0:
            logger.warning("per_category_target ignored: '{}' has negative value {}", raw_cat, target)
            continue

        canonical_cat = cat_lookup[key]
        resolved[canonical_cat] = target

    return resolved


def _allocate_with_overrides(
    category_pool_sizes: Dict[str, int],
    total_target: int,
    min_per_category: int,
    max_per_category: int,
    per_category_target: Dict[str, int],
) -> Dict[str, int]:
    """Allocate budget with optional fixed targets for selected categories.

    Selected categories in ``per_category_target`` are treated as fixed allocations
    (clamped to pool size). Remaining budget is distributed across other categories.
    """
    if not per_category_target:
        return _allocate_budget(category_pool_sizes, total_target, min_per_category, max_per_category)

    pools = dict(category_pool_sizes)
    allocation = {c: 0 for c in pools.keys()}
    fixed_categories = set()

    for cat, requested in per_category_target.items():
        if cat not in pools:
            continue
        fixed_categories.add(cat)
        allocation[cat] = min(max(0, int(requested)), pools[cat])

    fixed_total = sum(allocation[c] for c in fixed_categories)
    if fixed_total > total_target:
        raise ValueError(
            f"Sum of per_category_target ({fixed_total}) exceeds total_target ({total_target}). "
            "Lower per-category targets or increase total_target."
        )

    remaining_categories = [c for c in pools.keys() if c not in fixed_categories]
    remaining_target = total_target - fixed_total
    if not remaining_categories or remaining_target <= 0:
        return allocation

    remaining_pools = {c: pools[c] for c in remaining_categories}
    remaining_alloc = _allocate_budget(
        category_counts=remaining_pools,
        total_target=remaining_target,
        min_per_category=0,
        max_per_category=max_per_category,
    )

    for c, v in remaining_alloc.items():
        allocation[c] = v

    return allocation


@app.command()
def main(
    config_file: str = typer.Argument(..., help="Path to the asos_v1_sampling.yaml config"),
) -> None:
    """Sample images from the ASOS catalog using the combined filter spec."""
    cfg_path = _resolve(config_file)
    if not cfg_path.exists():
        logger.error("Config not found: {}", cfg_path)
        raise SystemExit(1)

    cfg = OmegaConf.load(str(cfg_path))

    # ── Load catalog ──────────────────────────────────────────────────────────
    catalog_csv = _resolve(str(cfg.catalog.csv))
    image_col = str(cfg.catalog.image_url_column)
    additional_col = str(cfg.catalog.get("additional_image_column", "") or "")
    category_col = str(cfg.catalog.get("category_column", "category"))
    gender_col = str(cfg.catalog.get("gender_column", "gender"))

    logger.info("Loading catalog from {}", catalog_csv)
    df = pd.read_csv(catalog_csv, dtype={"product_id": str})
    logger.info("Loaded {} products", len(df))

    # ── Expand to per-image rows ──────────────────────────────────────────────
    logger.info("Expanding products to individual image URLs…")
    expanded = _expand_images(df, image_col=image_col, additional_col=additional_col or None)
    logger.info("Expanded to {} image rows", len(expanded))

    # ── Apply filter spec ────────────────────────────────────────────────────
    filter_spec_path = _resolve(str(cfg.filter_spec))
    logger.info("Applying filter spec from {}", filter_spec_path)
    filtered = _apply_filter_spec(expanded, filter_spec_path, category_col=category_col, gender_col=gender_col)
    if filtered.empty:
        logger.error("No images passed the filter — aborting.")
        raise SystemExit(1)
    logger.info("Filter produced {} image rows across categories", len(filtered))

    # ── Budget allocation ────────────────────────────────────────────────────
    sampling_cfg = cfg.sampling
    total_target = int(sampling_cfg.total_target)
    min_per = int(sampling_cfg.get("min_per_category", 20))
    max_per = int(sampling_cfg.get("max_per_category", 400))
    seed = int(sampling_cfg.get("random_seed", 42))
    per_category_target_raw = dict(sampling_cfg.get("per_category_target", {}) or {})

    category_pool_sizes = filtered.groupby("category_clean")["im_url"].count().to_dict()
    logger.info("Categories in pool: {}", sorted(category_pool_sizes.keys()))

    per_category_target = _resolve_per_category_targets(
        raw_targets=per_category_target_raw,
        category_pool_sizes=category_pool_sizes,
    )
    if per_category_target:
        logger.info("Using per-category fixed targets: {}", per_category_target)

    allocation = _allocate_with_overrides(
        category_pool_sizes=category_pool_sizes,
        total_target=total_target,
        min_per_category=min_per,
        max_per_category=max_per,
        per_category_target=per_category_target,
    )
    logger.info(
        "Allocated {} images across {} categories (target={})",
        sum(allocation.values()),
        len(allocation),
        total_target,
    )

    # ── Sample per category ───────────────────────────────────────────────────
    sampled_frames: List[pd.DataFrame] = []
    stats_rows: List[Dict[str, Any]] = []

    for category, budget in sorted(allocation.items()):
        pool = filtered[filtered["category_clean"] == category]
        n = min(budget, len(pool))
        sample = pool.sample(n=n, random_state=seed).copy()
        sampled_frames.append(sample)
        stats_rows.append(
            {
                "category": category,
                "pool_size": len(pool),
                "allocated": budget,
                "sampled": n,
            }
        )
        logger.info("  {:<30s} pool={:>5d}  allocated={:>4d}  sampled={:>4d}", category, len(pool), budget, n)

    result = pd.concat(sampled_frames, ignore_index=True)
    result = result.drop_duplicates(subset=["im_url"]).reset_index(drop=True)

    # ── Keep only useful output columns ──────────────────────────────────────
    keep_cols = ["product_id", "category_clean", "matched_rule", "im_url"]
    for extra in ["gender", "title", "colour", "brand"]:
        if extra in result.columns:
            keep_cols.append(extra)
    result = result[[c for c in keep_cols if c in result.columns]]
    result = result.rename(columns={"category_clean": "category"})

    # ── Write outputs ─────────────────────────────────────────────────────────
    pool_csv = _resolve(str(cfg.output.pool_csv))
    stats_csv = _resolve(str(cfg.output.stats_csv))
    pool_csv.parent.makedirs_p()

    result.to_csv(pool_csv, index=False)
    logger.info("Wrote pool CSV ({} rows) → {}", len(result), pool_csv)

    stats_df = pd.DataFrame(stats_rows).sort_values("category")
    stats_df.to_csv(stats_csv, index=False)
    logger.info("Wrote stats CSV → {}", stats_csv)

    # Summary
    logger.info("\n{}", stats_df.to_string(index=False))
    logger.info("\nTotal sampled: {} images from {} categories", len(result), len(stats_rows))


if __name__ == "__main__":
    app()
