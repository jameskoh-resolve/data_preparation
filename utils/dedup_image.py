import hashlib
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from loguru import logger

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
