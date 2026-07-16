#!/usr/bin/env python
"""Gender-balanced sampler for the Westside human-present dataset.

Produces a ~3000 image sample that is approximately 70% women / 30% men,
includes ALL images from specified must-include categories, and randomly
samples the remainder to hit the target.

Usage:
    python scripts/curation/gender_sampler.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# --- Configuration ---
HUMAN_WORN_CSV = REPO_ROOT / "curated_datasets/TataTrent_Westside/categories/westside_human_worn_sample.csv"
ORIGINAL_CATALOG_CSV = REPO_ROOT / "datasets/catalogs/westside_catalog.csv"
OUTPUT_CSV = REPO_ROOT / "curated_datasets/TataTrent_Westside/categories/westside_gender_sample_3000.csv"

TARGET_MIN = 3000
WOMEN_RATIO = 0.70
MEN_RATIO = 0.30

# Categories where ALL images must be included (case-insensitive substring match)
MUST_INCLUDE_CATEGORIES = ["rings", "earrings", "bracelets", "formal trousers", "formal shirts"]

# Categories to exclude entirely
EXCLUDE_CATEGORIES = ["flip-flops", "co-ordinate sets"]

RANDOM_SEED = 42


def classify_gender(tags_str: str) -> str:
    if pd.isna(tags_str):
        return "unknown"
    try:
        tags = ast.literal_eval(tags_str)
    except Exception:
        return "unknown"
    tags_lower = set(t.lower() for t in tags)

    is_women = any(t in tags_lower for t in ["woman", "women", "female", "ladies", "lady"])
    is_men = any(t in tags_lower for t in ["man", "men", "male", "gents"])
    is_kids = any(t in tags_lower for t in ["kids", "boys", "boy"]) and not is_women and not is_men

    if is_kids:
        return "kids"
    if is_women and not is_men:
        return "women"
    if is_men and not is_women:
        return "men"
    if is_women and is_men:
        return "unisex"
    return "unknown"


def main() -> None:
    logger.info("Loading human-worn dataset: {}", HUMAN_WORN_CSV)
    human_df = pd.read_csv(HUMAN_WORN_CSV, dtype={"product_id": str})

    logger.info("Loading original catalog for tags: {}", ORIGINAL_CATALOG_CSV)
    orig = pd.read_csv(ORIGINAL_CATALOG_CSV, dtype={"product_id": str})
    orig_tags = orig[["product_id", "tags"]].drop_duplicates(subset="product_id")
    orig_tags.columns = ["product_id", "product_tags"]

    human_df["product_id"] = human_df["product_id"].astype(str)
    orig_tags["product_id"] = orig_tags["product_id"].astype(str)
    merged = human_df.merge(orig_tags, on="product_id", how="left")

    # Classify gender
    merged["gender"] = merged["product_tags"].apply(classify_gender)
    logger.info("Gender distribution before filtering:\n{}", merged["gender"].value_counts().to_string())

    # Normalize category for matching
    merged["_cat_lower"] = merged["category"].fillna("").astype(str).str.lower()

    # Exclude unwanted categories
    for exc in EXCLUDE_CATEGORIES:
        merged = merged[~merged["_cat_lower"].str.contains(exc, na=False)]
    logger.info("After excluding {}: {} rows", EXCLUDE_CATEGORIES, len(merged))

    # Separate must-include categories (all images kept)
    must_include_mask = merged["_cat_lower"].apply(
        lambda cat: any(term in cat for term in MUST_INCLUDE_CATEGORIES)
    )
    must_include_df = merged[must_include_mask].copy()
    remainder_df = merged[~must_include_mask].copy()

    logger.info("Must-include categories: {} images", len(must_include_df))
    logger.info("  Breakdown: {}", must_include_df["_cat_lower"].value_counts().to_dict())

    # Only keep women and men for the random sampling portion (exclude kids/unknown/unisex)
    remainder_women = remainder_df[remainder_df["gender"] == "women"].copy()
    remainder_men = remainder_df[remainder_df["gender"] == "men"].copy()

    # Calculate total needed to include all must-includes AND hit 70/30 ratio
    # with at least TARGET_MIN images total.
    must_women = len(must_include_df[must_include_df["gender"] == "women"])
    must_men = len(must_include_df[must_include_df["gender"] == "men"])
    must_other = len(must_include_df) - must_women - must_men

    # To hit 70/30, figure out what total is needed so that must-include fits
    # The larger constraint: must_men <= 0.30 * total OR must_women <= 0.70 * total
    # total >= must_men / MEN_RATIO and total >= must_women / WOMEN_RATIO
    implied_total_by_men = int(must_men / MEN_RATIO) if must_men > 0 else 0
    implied_total_by_women = int(must_women / WOMEN_RATIO) if must_women > 0 else 0
    target_total = max(TARGET_MIN, implied_total_by_men, implied_total_by_women)

    # Add must_other to budget since they don't count toward either ratio
    target_total += must_other

    target_women_total = int((target_total - must_other) * WOMEN_RATIO)
    target_men_total = int((target_total - must_other) * MEN_RATIO)

    need_women = max(0, target_women_total - must_women)
    need_men = max(0, target_men_total - must_men)

    logger.info("Target total: {} (min {} expanded to fit ratio)", target_total, TARGET_MIN)
    logger.info("Sampling: {} women + {} men from remainder pool", need_women, need_men)

    sampled_women = remainder_women.sample(n=min(need_women, len(remainder_women)), random_state=RANDOM_SEED)
    sampled_men = remainder_men.sample(n=min(need_men, len(remainder_men)), random_state=RANDOM_SEED)

    final = pd.concat([must_include_df, sampled_women, sampled_men], ignore_index=True)

    # Drop helper columns
    final = final.drop(columns=["_cat_lower"], errors="ignore")

    # Shuffle final output
    final = final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    # Report
    logger.info("Final dataset: {} images", len(final))
    logger.info("Gender split:\n{}", final["gender"].value_counts().to_string())
    logger.info("Category breakdown:\n{}", final["category"].value_counts().to_string())

    final.to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved to {}", OUTPUT_CSV)


if __name__ == "__main__":
    main()
