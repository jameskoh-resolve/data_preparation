import hashlib
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from loguru import logger
from path import Path

from utils.vis_image import get_image_content, ImageContentCache


def get_image_md5(url: str, cache: Optional[ImageContentCache] = None) -> str:
    """Fetch image bytes (from cache or URL) and calculate the MD5 hash."""
    try:
        if cache is not None:
            content = cache.get(url)
        else:
            content = get_image_content(url)
            
        if content:
            return hashlib.md5(content).hexdigest()
    except Exception as e:
        logger.error("Failed to compute hash for image {}: {}", url, e)
    return ""


def deduplicate_catalog_by_image(
    df: pd.DataFrame, 
    url_col: str = "im_url", 
    max_workers: int = 16,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Deduplicate a Pandas DataFrame containing image URLs by downloading the images
    in parallel, computing their MD5 hashes, and dropping duplicates.
    """
    if df.empty:
        return df

    urls = df[url_col].unique()
    logger.info("Computing content hashes for {} unique image URLs...", len(urls))
    
    cache = ImageContentCache(cache_dir) if cache_dir else ImageContentCache()

    # Process hashes in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        hashes = list(executor.map(lambda url: get_image_md5(url, cache=cache), urls))

    url_to_hash = dict(zip(urls, hashes))

    # Add hash column
    df_copy = df.copy()
    df_copy["image_content_hash"] = df_copy[url_col].map(url_to_hash)

    # Filter out failed downloads/hashes
    failed_count = sum(1 for h in hashes if not h)
    if failed_count > 0:
        logger.warning("Failed to download or compute hash for {} images.", failed_count)

    # Drop duplicates on the hash column
    initial_len = len(df_copy)
    df_deduped = df_copy[df_copy["image_content_hash"] != ""].drop_duplicates(
        subset=["image_content_hash"], keep="first"
    )
    
    deduped_count = initial_len - len(df_deduped)
    logger.info("Deduplication complete. Removed {} duplicate rows. Remaining: {}/{}", 
                deduped_count, len(df_deduped), initial_len)
                
    return df_deduped


def load_or_compute_exclusion_hashes(
    exclude_csv: str | Path, 
    exclude_col: str = "im_name", 
    max_workers: int = 16
) -> set[str]:
    """Load exclusion MD5 hashes from CSV, or compute & save them if missing/empty."""
    from path import Path
    exclude_csv_path = Path(exclude_csv)
    if not exclude_csv_path.exists():
        logger.warning("Exclusion CSV specified but not found: {}", exclude_csv_path)
        return set()

    logger.info("Loading exclusion dataset from {}...", exclude_csv_path)
    exclude_df = pd.read_csv(exclude_csv_path, dtype=str)

    # Check if exclude_col exists and has valid 32-char MD5 hashes
    valid_hashes = set()
    if exclude_col in exclude_df.columns:
        valid_hashes = {
            val for val in exclude_df[exclude_col].dropna().unique()
            if len(val) == 32 and all(c in "0123456789abcdefABCDEF" for c in val)
        }

    if valid_hashes:
        logger.info("Loaded {} valid image content MD5 hashes from column '{}'.", len(valid_hashes), exclude_col)
        return valid_hashes

    # Locate URL column to compute hashes
    url_col = None
    for c in ("im_url", "original_url", "image_url", "url"):
        if c in exclude_df.columns:
            url_col = c
            break

    if not url_col:
        logger.warning("No URL column found in {} to extract image hashes.", exclude_csv_path)
        return set()

    urls = [str(u).strip() for u in exclude_df[url_col].dropna().unique() if str(u).strip().startswith(("http://", "https://"))]
    logger.info("No pre-computed hashes in '{}' column. Computing image MD5 hashes for {} unique URLs...", exclude_col, len(urls))

    cache = ImageContentCache()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        hashes = list(executor.map(lambda u: get_image_md5(u, cache=cache), urls))

    url_to_hash = {u: h for u, h in zip(urls, hashes) if h}
    computed_hashes = set(url_to_hash.values())
    logger.info("Successfully extracted {} unique image content MD5 hashes.", len(computed_hashes))

    # Persist computed hashes back to CSV
    if "im_name" not in exclude_df.columns:
        exclude_df["im_name"] = ""
    
    exclude_df["im_name"] = exclude_df[url_col].map(url_to_hash).fillna(exclude_df.get(exclude_col, ""))
    if exclude_col != "im_name" and exclude_col in exclude_df.columns:
        exclude_df[exclude_col] = exclude_df["im_name"]

    exclude_df.to_csv(exclude_csv_path, index=False)
    logger.info("Updated and saved computed hashes back to {}", exclude_csv_path)

    return computed_hashes

