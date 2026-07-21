#!/usr/bin/env python
"""Class sampler script for Selectedhomme active sampling.

Reads catalog, shuffles products, filters to front-facing model images using
the virecognition tagging API, and verifies target accessories using LLM.
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
import hashlib
from typing import Any, Dict, List, Optional

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
from pydantic import BaseModel, Field
from tqdm import tqdm

app = typer.Typer(pretty_exceptions_show_locals=False)

REPO_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.vis_image import ImageContentCache, get_image_content
from llm.executor import LLMExecutor
from llm.gpt_anno_helpers import BaseAnnotator, annotate_df as llm_annotate_df

load_dotenv(dotenv_path=Path(os.environ["HOME"]) / ".dltk.config")

VIRECOG_API_HOST = "https://virecognition.visenze.com/v1/image/recognize"
VIRECOG_AUTH = HTTPBasicAuth(os.environ["DP_ACCESS_KEY"], os.environ["DP_SECRET_KEY"])

IDENTIFY_CLASSES_SYSTEM_PATH = REPO_ROOT / "llm" / "prompts" / "curate_dataset_one_pass_system.txt"
IDENTIFY_CLASSES_PROMPT_PATH = REPO_ROOT / "llm" / "prompts" / "identify_classes_prompt_v4.txt"


class IdentifyClassesResult(BaseModel):
    visible_classes: List[str] = Field(default_factory=list, description="Target accessory classes visible in the image")
    gpt_reason: str = Field(..., description="Brief reasoning for the decision")


class IdentifyClassesAnnotator(BaseAnnotator):
    annotation_output_fields = ["visible_classes", "gpt_reason"]
    annotation_output_object = IdentifyClassesResult
    prompt_path = IDENTIFY_CLASSES_PROMPT_PATH

    def __init__(self, model_name: str = "gpt-4.1-mini-2025-04-14", prod_type: str = "fashion"):
        super().__init__(system_path=IDENTIFY_CLASSES_SYSTEM_PATH, model_name=model_name, prod_type=prod_type)
        self.im_content_cache = ImageContentCache()
        self._executor = LLMExecutor.from_model_name(model_name, image_content_cache=self.im_content_cache)

    def process_row(self, row: pd.Series) -> dict:
        im_url = row["im_url"]
        im_source = row.get("llm_image_source", im_url)
        available_classes = row.get("available_classes", "")

        im_content = self.im_content_cache.get(im_source)
        if im_content is None:
            logger.warning("Failed to fetch image content for source {}, returning empty classes", im_source)
            return {
                "visible_classes": "[]",
                "gpt_reason": "image fetch failed",
            }

        prompt_messages = self.chat_prompt.format_prompt(
            available_classes=available_classes,
        ).to_messages()

        parsed = self._executor.predict(
            prompt_messages,
            images=[im_content],
            output_object_type=self.annotation_output_object,
        )

        visible_classes = [str(x).strip().lower() for x in (parsed.visible_classes or []) if str(x).strip()]
        return {
            "visible_classes": json.dumps(visible_classes, ensure_ascii=True),
            "gpt_reason": parsed.gpt_reason,
        }


def _resolve_path(path_value: str) -> Path:
    path_obj = Path(path_value)
    if path_obj.isabs():
        return path_obj
    return REPO_ROOT / path_obj


def _parse_list_field(cell: Any) -> List[str]:
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


def _stable_image_id(catalog_name: str, product_id: str, image_url: str) -> str:
    digest = hashlib.md5(f"{catalog_name}|{product_id}|{image_url}".encode("utf-8")).hexdigest()[:16]
    return digest


def _recognize_single_image(
    im_url: str,
    tag_group_name: str,
    tag_group_version: str,
    exclude_hashes: set[str] = None,
) -> tuple[str, list[str], str]:
    content = get_image_content(im_url)
    if content is None:
        logger.warning("Failed to fetch image {}, skipping", im_url)
        return im_url, [], ""

    md5_hash = hashlib.md5(content).hexdigest()
    if exclude_hashes and md5_hash in exclude_hashes:
        logger.info("Image {} hash {} is in exclude list, skipping API", im_url, md5_hash)
        return im_url, ["deduplicated"], md5_hash

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
        return im_url, [], md5_hash

    if result.get("status") != "OK" or not result.get("result"):
        logger.warning("virecognition returned non-OK for {}: {}", im_url, result.get("error", result.get("status")))
        return im_url, [], md5_hash

    all_tags: list[str] = []
    for tag_result in result["result"]:
        for obj in tag_result.get("objects", []):
            for tag in obj.get("tags", []):
                tag_name = tag.get("tag", "")
                if tag_name:
                    all_tags.append(tag_name)
    return im_url, sorted(all_tags), md5_hash


def _tag_images(
    batch_df: pd.DataFrame,
    model_config_path: str,
    n_jobs: int,
    exclude_hashes: set[str],
) -> tuple[pd.DataFrame, list[str]]:
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)
    tag_group_name = model_cfg["name"]
    tag_group_version = str(model_cfg.get("mlp_version", model_cfg.get("version", "")))

    unique_urls = batch_df["im_url"].drop_duplicates().tolist()
    logger.info("Tagging {} unique images via virecognition API...", len(unique_urls))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_recognize_single_image)(url, tag_group_name, tag_group_version, exclude_hashes)
        for url in unique_urls
    )

    url_to_tags: dict[str, str] = {}
    new_hashes: list[str] = []
    
    for url, tags, md5_hash in results:
        url_to_tags[url] = ",".join(tags)
        if md5_hash:
            new_hashes.append(md5_hash)
            
    out = batch_df.copy()
    out["tags"] = out["im_url"].map(url_to_tags).fillna("")
    return out, new_hashes


def _expand_catalog(catalog_cfg: DictConfig, catalog_name: str, seed: int) -> pd.DataFrame:
    catalog_csv = _resolve_path(str(catalog_cfg.csv))
    image_col = str(catalog_cfg.image_url_column)
    additional_col = str(catalog_cfg.get("additional_image_column", "") or "")
    category_col = str(catalog_cfg.get("category_column", "category"))

    logger.info("Loading catalog from {}", catalog_csv)
    df = pd.read_csv(catalog_csv, dtype={"product_id": str})
    logger.info("Loaded {} products", len(df))

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
                row = {
                    "catalog_name": catalog_name,
                    "product_id": str(record.get("product_id", "")),
                    "title": str(record.get("title", "")),
                    "brand": str(record.get("brand", "")),
                    "colour": str(record.get("colors", record.get("colour", ""))),
                    "category": str(record.get(category_col, "")),
                    "im_url": url,
                    "im_id": _stable_image_id(catalog_name, str(record.get("product_id", "")), url),
                    "tags": str(record.get("tags", "")),
                }
                rows.append(row)

    expanded_df = pd.DataFrame(rows)
    logger.info("Expanded catalog to {} image rows", len(expanded_df))

    # Shuffle to prevent category or order skew
    shuffled_df = expanded_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info("Shuffled candidate rows.")
    return shuffled_df


@app.command()
def main(
    config_file: str = typer.Argument(..., help="Path to class sampler config YAML"),
) -> None:
    cfg_path = _resolve_path(config_file)
    if not cfg_path.exists():
        logger.error("Config not found: {}", cfg_path)
        raise SystemExit(1)

    cfg = OmegaConf.load(str(cfg_path))
    seed = int(cfg.run.seed)
    target_size = int(cfg.run.target_size)
    batch_size = int(cfg.run.get("batch_size", 2000))
    output_dir = _resolve_path(str(cfg.run.output_dir))
    output_dir.makedirs_p()

    catalog_name = str(cfg.run.get("catalog_name", "selectedhomme_v1"))
    skip_tagging = bool(cfg.run.get("skip_tagging", False))
    
    # 1. Load and Shuffle Catalog
    candidates = _expand_catalog(cfg.catalog, catalog_name, seed)

    # Setup configurations
    tagging_cfg = cfg.get("tagging", {})
    model_config_path = str(tagging_cfg.get("model_config", REPO_ROOT / "configs/tagging_models/fashion_eval_image.yaml"))
    n_jobs = int(tagging_cfg.get("n_jobs", 16))

    llm_cfg = cfg.get("llm", {})
    model_name = str(llm_cfg.get("model", "gpt-4.1-mini-2025-04-14"))
    workers = int(llm_cfg.get("workers", 8))
    checkpoint_root = _resolve_path(str(llm_cfg.get("checkpoint_dir", str(output_dir / "llm_anno"))))
    checkpoint_dir = checkpoint_root / "identify_classes"

    relevant_classes = set(cfg.classes.relevant_classes)
    available_classes = ", ".join(sorted(relevant_classes))

    # Must-include categories bypass the front-facing filter entirely
    must_include_categories = [str(x).strip().lower() for x in (cfg.run.get("must_include_categories", []) or [])]

    accepted_rows: List[pd.DataFrame] = []
    rejected_rows: List[pd.DataFrame] = []

    # Pre-accept must-include category images after running tagging and LLM verification on them
    if must_include_categories:
        cat_lower = candidates["category"].fillna("").astype(str).str.lower()
        must_mask = cat_lower.apply(lambda c: any(term in c for term in must_include_categories))
        must_include_df = candidates[must_mask].copy()
        candidates = candidates[~must_mask].reset_index(drop=True)
        if not must_include_df.empty:
            # Cap to target_size to prevent wasteful LLM calls
            must_include_df = must_include_df.head(target_size)
            logger.info("Pre-accepting and running LLM verification on {} must-include images...", len(must_include_df))
            
            # Step A: Fast API Tagging Filter
            if skip_tagging and "tags" in must_include_df.columns:
                must_include_tagged = must_include_df.copy()
                logger.info("Skipping virecognition tagging for must-include images (tags already present in input).")
            else:
                must_include_tagged, _ = _tag_images(must_include_df, model_config_path=model_config_path, n_jobs=n_jobs, exclude_hashes=set())
            
            # Step B: LLM Accessory Verification
            must_include_anno = must_include_tagged.copy()
            must_include_anno["available_classes"] = available_classes
            must_include_anno["llm_image_source"] = must_include_anno["im_url"].astype(str)
            
            annotator = IdentifyClassesAnnotator(model_name=model_name, prod_type="fashion")
            must_include_annotated = llm_annotate_df(
                df=must_include_anno,
                annotation_keys=["im_id"],
                annotator=annotator,
                checkpoint_dir=checkpoint_dir,
                checkpoint_interval=10,
                max_workers=workers,
            )
            
            # Merge results back
            anno_keep_cols = ["im_id", "visible_classes", "gpt_reason"]
            annotated_result = must_include_annotated[anno_keep_cols].drop_duplicates(subset=["im_id"], keep="last")
            must_include_tagged = must_include_tagged.merge(annotated_result, on="im_id", how="left")
            must_include_tagged["visible_classes"] = must_include_tagged["visible_classes"].fillna("[]")
            must_include_tagged["gpt_reason"] = must_include_tagged["gpt_reason"].fillna("llm_failed")
            
            # For accepted must-include images, append detected accessories to 'tags' column for the visualizer
            def merge_tags_and_accessories(row):
                orig_tags = [t.strip() for t in row["tags"].split(",") if t.strip()]
                try:
                    acc_classes = json.loads(row["visible_classes"])
                except:
                    acc_classes = []
                combined = orig_tags + [c for c in acc_classes if c not in orig_tags]
                return ",".join(combined)
            
            must_include_tagged["tags"] = must_include_tagged.apply(merge_tags_and_accessories, axis=1)
            accepted_rows.append(must_include_tagged)
            logger.info("Successfully pre-accepted {} must-include images with LLM classes.", len(must_include_tagged))

    candidate_idx = 0
    round_count = 0
    accepted_count = 0
    exclude_hashes: set[str] = set()

    while accepted_count < target_size and candidate_idx < len(candidates):
        round_count += 1
        missing = target_size - accepted_count
        current_batch_size = min(batch_size, len(candidates) - candidate_idx)
        if current_batch_size <= 0:
            break

        logger.info("--- Round {} | Candidates Checked: {}/{} | Accepted: {}/{} ---", 
                    round_count, candidate_idx, len(candidates), accepted_count, target_size)

        # Get batch of raw candidates
        batch = candidates.iloc[candidate_idx : candidate_idx + current_batch_size].reset_index(drop=True)
        candidate_idx += current_batch_size

        # Step A: Fast API Tagging Filter
        if skip_tagging and "tags" in batch.columns:
            tagged = batch.copy()
            logger.info("Skipping virecognition tagging (tags already present in input).")
        else:
            tagged, batch_hashes = _tag_images(batch, model_config_path=model_config_path, n_jobs=n_jobs, exclude_hashes=exclude_hashes)
            exclude_hashes.update(batch_hashes)
        
        # Keep only frontward facing human models
        def passes_tags(tags_str: str) -> bool:
            tags = [t.strip() for t in tags_str.split(",") if t.strip()]
            has_human = "image_human:model" in tags
            front_facing = "human_angle:front_view" in tags or "human_angle:front_angled_view" in tags
            return has_human and front_facing

        front_facing_batch = tagged[tagged["tags"].apply(passes_tags)].copy()
        not_front_facing = tagged[~tagged["tags"].apply(passes_tags)].copy()

        if not not_front_facing.empty:
            not_front_facing["reject_reason"] = "not_front_facing_model"
            rejected_rows.append(not_front_facing)

        if front_facing_batch.empty:
            logger.info("0 front-facing model images in this batch. Continuing...")
            continue

        logger.info("Found {} frontward-facing model images. Running LLM verification...", len(front_facing_batch))

        # Step B: LLM Accessory Verification
        anno_rows = front_facing_batch.copy()
        anno_rows["available_classes"] = available_classes
        anno_rows["llm_image_source"] = anno_rows["im_url"].astype(str)

        annotator = IdentifyClassesAnnotator(model_name=model_name, prod_type="fashion")
        annotated = llm_annotate_df(
            df=anno_rows,
            annotation_keys=["im_id"],
            annotator=annotator,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=10,
            max_workers=workers,
        )

        # Merge results back
        anno_keep_cols = ["im_id", "visible_classes", "gpt_reason"]
        annotated_result = annotated[anno_keep_cols].drop_duplicates(subset=["im_id"], keep="last")
        front_facing_batch = front_facing_batch.merge(annotated_result, on="im_id", how="left")
        front_facing_batch["visible_classes"] = front_facing_batch["visible_classes"].fillna("[]")
        front_facing_batch["gpt_reason"] = front_facing_batch["gpt_reason"].fillna("llm_failed")

        # Check for accessory presence
        def has_relevant_accessory(vis_classes_str: str) -> bool:
            try:
                classes = json.loads(vis_classes_str)
                return any(c in relevant_classes for c in classes)
            except:
                return False

        positives = front_facing_batch[front_facing_batch["visible_classes"].apply(has_relevant_accessory)].copy()
        negatives = front_facing_batch[~front_facing_batch["visible_classes"].apply(has_relevant_accessory)].copy()

        # For accepted positives, append detected accessories to 'tags' column for the visualizer
        if not positives.empty:
            def merge_tags_and_accessories(row):
                orig_tags = [t.strip() for t in row["tags"].split(",") if t.strip()]
                try:
                    acc_classes = json.loads(row["visible_classes"])
                except:
                    acc_classes = []
                # Keep original tags and append accessory classes
                combined = orig_tags + [c for c in acc_classes if c not in orig_tags]
                return ",".join(combined)

            positives["tags"] = positives.apply(merge_tags_and_accessories, axis=1)
            accepted_rows.append(positives)
            accepted_count = len(pd.concat(accepted_rows, ignore_index=True))

        if not negatives.empty:
            negatives["reject_reason"] = "no_visible_accessories"
            rejected_rows.append(negatives)

    # Wrap up and Save Results
    accepted_df = pd.concat(accepted_rows, ignore_index=True) if accepted_rows else pd.DataFrame()
    rejected_df = pd.concat(rejected_rows, ignore_index=True) if rejected_rows else pd.DataFrame()

    if not accepted_df.empty:
        # Deduplicate and crop to target_size
        accepted_df = accepted_df.drop_duplicates(subset=["im_url"]).head(target_size)
        accepted_df.to_csv(output_dir / "human_worn_sample.csv", index=False)
        accepted_df.to_csv(output_dir / f"{catalog_name}_human_worn_sample.csv", index=False)
        logger.info("Saved {} positive samples to {}/human_worn_sample.csv", len(accepted_df), output_dir)

    if not rejected_df.empty:
        rejected_df.to_csv(output_dir / "human_worn_rejected.csv", index=False)

    stats = {
        catalog_name: {
            "total_candidates": int(len(candidates)),
            "checked": int(candidate_idx),
            "accepted": int(len(accepted_df)) if not accepted_df.empty else 0,
            "target": int(target_size),
            "rounds": int(round_count),
            "exhausted": bool(candidate_idx >= len(candidates)),
        }
    }
    (output_dir / "human_worn_sampling_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Sampling completed.")


if __name__ == "__main__":
    app()
