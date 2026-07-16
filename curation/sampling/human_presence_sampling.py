#!/usr/bin/env python
"""Standalone human-presence sampler for catalog images using the fashion_eval_image API model.

The script samples catalog images, calls the fashion_eval_image virecognition model
to determine whether a human is present in each image, and keeps resampling until a
target number of accepted images is reached per catalog.

Requires DP_ACCESS_KEY and DP_SECRET_KEY to be set in ~/.dltk.config.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

import pandas as pd
import requests
import typer
import yaml
from dotenv import load_dotenv
from joblib import Parallel, delayed
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from path import Path
from requests.auth import HTTPBasicAuth

app = typer.Typer(pretty_exceptions_show_locals=False)

REPO_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.vis_image import get_image_content

load_dotenv(dotenv_path=Path(os.environ["HOME"]) / ".dltk.config")

DEFAULT_MODEL_CONFIG = "fashion_eval_image.yaml"
VIRECOG_API_HOST = "https://virecognition.visenze.com/v1/image/recognize"
VIRECOG_AUTH = HTTPBasicAuth(os.environ["DP_ACCESS_KEY"], os.environ["DP_SECRET_KEY"])


def _resolve_path(path_value: str) -> Path:
    path_obj = Path(path_value)
    if path_obj.isabs():
        return path_obj
    return REPO_ROOT / path_obj


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _column_as_str(df: pd.DataFrame, column_name: str) -> pd.Series:
    if column_name in df.columns:
        return df[column_name].astype(str)
    return pd.Series([""] * len(df), index=df.index, dtype="object")


def _tag_list(tag_cell: Any) -> List[str]:
    text = str(tag_cell or "").strip()
    if not text:
        return []
    return [x.strip() for x in text.split(",") if x.strip()]


def _apply_detail_view_filter(batch_df: pd.DataFrame, catalog_cfg: DictConfig) -> pd.DataFrame:
    """Reject detail-view images for non-accessory categories.

    Rule:
    - If image has tag 'image_detail:detail', keep it only when category matches
      one of allow_detail_category_terms.
    - Non-detail images always pass this filter.
    """
    detail_cfg = catalog_cfg.get("detail_view_filter", {})
    enabled = _as_bool(detail_cfg.get("enabled", False), False)
    if not enabled:
        out = batch_df.copy()
        out["is_detail_view"] = False
        out["detail_allowed_category"] = True
        out["passes_detail_filter"] = True
        return out

    allow_terms = [str(x).strip().lower() for x in (detail_cfg.get("allow_detail_category_terms", []) or [])]
    allow_terms = [x for x in allow_terms if x]
    if not allow_terms:
        raise ValueError(
            "detail_view_filter.enabled is true but allow_detail_category_terms is empty. "
            "Configure allowed category terms in YAML."
        )

    out = batch_df.copy()
    out["_tag_list"] = out.get("tags", pd.Series([""] * len(out))).apply(_tag_list)
    out["is_detail_view"] = out["_tag_list"].apply(lambda tags: "image_detail:detail" in tags)
    out["_category_lc"] = out.get("category", pd.Series([""] * len(out))).astype(str).str.lower()
    out["detail_allowed_category"] = out["_category_lc"].apply(
        lambda cat: any(term in cat for term in allow_terms)
    )
    out["passes_detail_filter"] = (~out["is_detail_view"]) | out["detail_allowed_category"]
    out = out.drop(columns=["_tag_list", "_category_lc"])
    return out


def _parse_url_list(cell_value: Any) -> List[str]:
    if cell_value is None:
        return []
    if isinstance(cell_value, list):
        raw = cell_value
    else:
        text = str(cell_value).strip()
        if not text:
            return []
        raw = None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                raw = parsed
        except Exception:
            raw = None
        if raw is None:
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, list):
                    raw = parsed
            except Exception:
                raw = None
        if raw is None:
            return []

    out: List[str] = []
    for item in raw:
        url = str(item).strip()
        if url.startswith(("http://", "https://")):
            out.append(url)
    return out


def _stable_image_id(catalog_name: str, product_id: str, image_url: str) -> str:
    digest = hashlib.md5(f"{catalog_name}|{product_id}|{image_url}".encode("utf-8")).hexdigest()[:16]
    return digest


def _resolve_catalog_csv(catalog_cfg: DictConfig) -> Path:
    if "catalog_csv" in catalog_cfg and str(catalog_cfg.catalog_csv).strip():
        return _resolve_path(str(catalog_cfg.catalog_csv))

    if "catalog_dir" not in catalog_cfg:
        raise ValueError("Each catalog item must define either catalog_csv or catalog_dir")

    catalog_dir = _resolve_path(str(catalog_cfg.catalog_dir))
    if not catalog_dir.exists() or not catalog_dir.isdir():
        raise ValueError(f"catalog_dir does not exist or is not a directory: {catalog_dir}")

    preferred = catalog_dir / "catalog.csv"
    if preferred.exists():
        return preferred

    candidates = sorted([x for x in catalog_dir.files("*.csv") if x.isfile()])
    if not candidates:
        raise ValueError(f"No CSV found in catalog_dir: {catalog_dir}")
    return candidates[0]


def _filter_by_categories(df: pd.DataFrame, category_col: str, sample_categories: List[str]) -> pd.DataFrame:
    if not sample_categories:
        return df
    if category_col not in df.columns:
        logger.warning("Configured sample_categories but column '{}' is missing; skipping filter.", category_col)
        return df

    terms = [str(x).strip().lower() for x in sample_categories if str(x).strip()]
    if not terms:
        return df

    cat_series = df[category_col].fillna("").astype(str).str.lower()
    mask = cat_series.apply(lambda value: any(term in value for term in terms))
    return df[mask].copy()


def _load_catalog_pool(catalog_cfg: DictConfig) -> pd.DataFrame:
    catalog_name = str(catalog_cfg.name)
    catalog_csv = _resolve_catalog_csv(catalog_cfg)
    image_col = str(catalog_cfg.image_url_column)
    category_col = str(catalog_cfg.get("category_column", "category"))
    sample_categories = list(catalog_cfg.get("sample_categories", []) or [])
    include_additional_images = _as_bool(catalog_cfg.get("include_additional_images", False), False)
    additional_image_col = str(catalog_cfg.get("additional_image_column", "additional_image_url"))
    max_images_per_product = int(catalog_cfg.get("max_images_per_product", 0) or 0)

    logger.info("Loading catalog {} from {}", catalog_name, catalog_csv)
    df = pd.read_csv(catalog_csv, dtype={"product_id": str})
    if image_col not in df.columns:
        raise ValueError(f"Catalog {catalog_name} missing image column: {image_col}")

    work = df.copy()
    work["im_url"] = work[image_col].astype(str).str.strip()
    work = work[work["im_url"].str.len() > 0]
    work = work[work["im_url"].str.startswith(("http://", "https://"))]
    work = _filter_by_categories(work, category_col=category_col, sample_categories=sample_categories)

    if include_additional_images and additional_image_col in work.columns:
        expanded_rows: List[Dict[str, Any]] = []
        for row in work.to_dict("records"):
            main_url = str(row.get("im_url", "")).strip()
            additional_urls = _parse_url_list(row.get(additional_image_col, None))
            all_urls: List[str] = []
            for candidate in [main_url] + additional_urls:
                if candidate and candidate not in all_urls:
                    all_urls.append(candidate)

            if max_images_per_product > 0:
                all_urls = all_urls[:max_images_per_product]

            for url in all_urls:
                new_row = dict(row)
                new_row["im_url"] = url
                expanded_rows.append(new_row)

        if expanded_rows:
            work = pd.DataFrame(expanded_rows)

    work = work.drop_duplicates(subset=["im_url"]).reset_index(drop=True)
    work["catalog_name"] = catalog_name
    work["product_id"] = _column_as_str(work, "product_id")
    work["im_name"] = _column_as_str(work, "title")
    work["im_description"] = _column_as_str(work, "description")
    work["category"] = _column_as_str(work, category_col)
    work["im_id"] = work.apply(
        lambda row: _stable_image_id(catalog_name, str(row["product_id"]), str(row["im_url"])), axis=1
    )
    return work[["catalog_name", "product_id", "category", "im_url", "im_id", "im_name", "im_description"]].copy()


def _recognize_single_image(
    im_url: str,
    tag_group_name: str,
    tag_group_version: str,
) -> tuple[str, bool, list[str]]:
    """Download one image and call the virecognition API directly.

    Returns (im_url, has_human_evidence, tags) where tags is the full list of
    tag strings returned by the model (e.g. ['image_human:model', 'human_angle:front_view']).
    """
    content = get_image_content(im_url)
    if content is None:
        logger.warning("Failed to fetch image {}, skipping", im_url)
        return im_url, False, []

    files = [
        ("object_limit", (None, "1")),
        ("vtt_source", (None, "visenze_data_pipeline")),
        ("tag_group", (None, tag_group_name)),
        ("version", (None, f"{tag_group_name}:{tag_group_version}")),
        ("file", ("image.jpg", content, "image/jpeg")),
    ]
    try:
        resp = requests.post(VIRECOG_API_HOST, files=files, auth=VIRECOG_AUTH, timeout=120)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        logger.warning("virecognition API call failed for {}: {}", im_url, e)
        return im_url, False, []

    if result.get("status") != "OK" or not result.get("result"):
        logger.warning("virecognition returned non-OK for {}: {}", im_url, result.get("error", result.get("status")))
        return im_url, False, []

    all_tags: list[str] = []
    has_human = False
    for tag_result in result["result"]:
        for obj in tag_result.get("objects", []):
            for tag in obj.get("tags", []):
                tag_name = tag.get("tag", "")
                if tag_name:
                    all_tags.append(tag_name)
                if tag_name == "image_human:model":
                    has_human = True
    return im_url, has_human, sorted(all_tags)


def _tag_human_presence(
    batch_df: pd.DataFrame,
    model_config_path: str,
    n_jobs: int,
) -> pd.DataFrame:
    """Tag images using the virecognition API directly and set has_human_evidence column.

    Downloads images (with browser headers for ASOS) and POSTs them directly
    to the virecognition API via the 'file' multipart field, bypassing VIS upload.

    Returns batch_df with an added 'has_human_evidence' boolean column.
    """
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)
    tag_group_name = model_cfg["name"]
    tag_group_version = str(model_cfg.get("mlp_version", model_cfg.get("version", "")))

    unique_urls = batch_df["im_url"].drop_duplicates().tolist()
    logger.info("Tagging {} unique images via virecognition API...", len(unique_urls))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_recognize_single_image)(url, tag_group_name, tag_group_version)
        for url in unique_urls
    )

    url_to_human: dict[str, bool] = {}
    url_to_tags: dict[str, str] = {}
    for url, has_human, tags in results:
        url_to_human[url] = has_human
        url_to_tags[url] = ",".join(tags)
    out = batch_df.copy()
    out["has_human_evidence"] = out["im_url"].map(url_to_human).fillna(False)
    out["tags"] = out["im_url"].map(url_to_tags).fillna("")
    return out


def _run_sampling(
    cfg: DictConfig,
    target_per_catalog: int,
    output_dir: Path,
    oversample_factor: int,
    max_rounds: int,
    model_override: str | None,
) -> None:
    output_dir.makedirs_p()

    tagging_cfg = cfg.get("tagging", {})
    model_config_path = model_override or str(tagging_cfg.get("model_config", REPO_ROOT / DEFAULT_MODEL_CONFIG))
    n_jobs = int(tagging_cfg.get("n_jobs", 16))

    all_accepted: List[pd.DataFrame] = []
    all_rejected: List[pd.DataFrame] = []
    all_evaluated: List[pd.DataFrame] = []
    stats: Dict[str, Any] = {}

    for catalog_cfg in cfg.catalogs:
        catalog_name = str(catalog_cfg.name)
        pool = _load_catalog_pool(catalog_cfg)
        if pool.empty:
            stats[catalog_name] = {
                "pool_size": 0,
                "evaluated": 0,
                "accepted": 0,
                "target": target_per_catalog,
                "rounds": 0,
                "exhausted": True,
            }
            continue

        accepted_rows: List[pd.DataFrame] = []
        rejected_rows: List[pd.DataFrame] = []
        evaluated_rows: List[pd.DataFrame] = []
        remaining = pool.copy()
        round_count = 0
        accepted_count = 0

        while accepted_count < target_per_catalog and not remaining.empty and round_count < max_rounds:
            round_count += 1
            missing = target_per_catalog - accepted_count
            batch_size = min(len(remaining), max(missing, missing * oversample_factor))
            batch = remaining.sample(n=batch_size, random_state=round_count).reset_index(drop=True)

            annotated = _tag_human_presence(batch, model_config_path=model_config_path, n_jobs=n_jobs)
            if annotated.empty:
                break

            annotated = _apply_detail_view_filter(annotated, catalog_cfg)

            annotated = annotated.copy()
            annotated["round"] = round_count
            evaluated_rows.append(annotated)

            pos = annotated[
                (annotated["has_human_evidence"] == True)
                & (annotated["passes_detail_filter"] == True)
            ].copy()
            neg = annotated[
                (annotated["has_human_evidence"] == False)
                | (annotated["passes_detail_filter"] == False)
            ].copy()
            if not pos.empty:
                accepted_rows.append(pos)
                accepted_count = len(pd.concat(accepted_rows, ignore_index=True))
            if not neg.empty:
                rejected_rows.append(neg)

            remaining = remaining[~remaining["im_url"].isin(annotated["im_url"].unique())].copy()

        accepted_df = pd.concat(accepted_rows, ignore_index=True) if accepted_rows else pd.DataFrame()
        rejected_df = pd.concat(rejected_rows, ignore_index=True) if rejected_rows else pd.DataFrame()
        evaluated_df = pd.concat(evaluated_rows, ignore_index=True) if evaluated_rows else pd.DataFrame()

        if not accepted_df.empty:
            accepted_df = accepted_df.drop_duplicates(subset=["im_url"]).head(target_per_catalog)
            accepted_df.to_csv(output_dir / f"{catalog_name}_human_worn_sample.csv", index=False)
            all_accepted.append(accepted_df)
        if not rejected_df.empty:
            all_rejected.append(rejected_df)
        if not evaluated_df.empty:
            all_evaluated.append(evaluated_df)

        stats[catalog_name] = {
            "pool_size": int(len(pool)),
            "evaluated": int(len(evaluated_df)),
            "accepted": int(len(accepted_df)),
            "target": int(target_per_catalog),
            "rounds": int(round_count),
            "exhausted": bool(remaining.empty),
        }

    final_accepted = pd.concat(all_accepted, ignore_index=True) if all_accepted else pd.DataFrame()
    final_rejected = pd.concat(all_rejected, ignore_index=True) if all_rejected else pd.DataFrame()
    final_evaluated = pd.concat(all_evaluated, ignore_index=True) if all_evaluated else pd.DataFrame()

    final_accepted.to_csv(output_dir / "human_worn_sample.csv", index=False)
    final_rejected.to_csv(output_dir / "human_worn_rejected.csv", index=False)
    final_evaluated.to_csv(output_dir / "human_worn_all_evaluated.csv", index=False)
    (output_dir / "human_worn_sampling_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Saved accepted={}, rejected={}, evaluated={}", len(final_accepted), len(final_rejected), len(final_evaluated))


@app.command()
def main(
    config_file: str = typer.Argument(..., help="Path to curation config YAML"),
    target_per_catalog: int = typer.Option(50, help="Target number of accepted samples per catalog"),
    output_dir: str = typer.Option("curation/sample", help="Output directory for results"),
    oversample_factor: int = typer.Option(3, help="Oversample factor for each round"),
    max_rounds: int = typer.Option(10, help="Maximum rounds per catalog"),
    model_config: str = typer.Option("", help="Optional path to tagging model YAML config override"),
) -> None:
    cfg_path = _resolve_path(config_file)
    cfg = OmegaConf.load(str(cfg_path))

    sampling_cfg = cfg.get("sampling", {})
    resolved_target = int(sampling_cfg.get("target_per_catalog", cfg.get("target_per_catalog", target_per_catalog)))
    resolved_output_dir = str(sampling_cfg.get("output_dir", cfg.get("output_dir", output_dir)))
    resolved_oversample = int(sampling_cfg.get("oversample_factor", cfg.get("oversample_factor", oversample_factor)))
    resolved_max_rounds = int(sampling_cfg.get("max_rounds", cfg.get("max_rounds", max_rounds)))

    _run_sampling(
        cfg=cfg,
        target_per_catalog=resolved_target,
        output_dir=_resolve_path(resolved_output_dir),
        oversample_factor=resolved_oversample,
        max_rounds=resolved_max_rounds,
        model_override=model_config.strip() or None,
    )


if __name__ == "__main__":
    app()