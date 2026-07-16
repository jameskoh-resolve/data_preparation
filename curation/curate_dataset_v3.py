#!/usr/bin/env python
"""Curate dataset V3 script for Selectedhomme issue classification.

Runs model 1302 detections on the 2000 sampled images, reads LLM predictions
directly from human_worn_sample.csv, and classifies precision/recall issues.
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List

import pandas as pd
import typer
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from path import Path

app = typer.Typer(pretty_exceptions_show_locals=False)

REPO_ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detection.geometry import compute_iou
from detection.parallel_detect import ParallelDetect

STL_COLUMNS = [
    "original_url",
    "im_url",
    "im_id",
    "crop_box",
    "crop_box_concept",
    "im_name",
    "im_description",
    "labels",
    "concepts",
    "boxes",
    "optional",
]

EXTRA_COLUMNS = [
    "box_statuses",
    "box_sources",
    "box_reasons",
    "issue_types",
]

OUTPUT_COLUMNS = STL_COLUMNS + EXTRA_COLUMNS


def _resolve_path(path_value: str) -> Path:
    path_obj = Path(path_value)
    if path_obj.isabs():
        return path_obj
    return REPO_ROOT / path_obj


def _parse_class_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    else:
        text = str(value).strip()
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
            return [x.strip().lower() for x in text.split(",") if x.strip()]

    return [str(item).strip().lower() for item in raw if str(item).strip()]


def _load_and_sample_catalogs(cfg: DictConfig) -> pd.DataFrame:
    catalog_cfg = cfg.catalogs[0]
    catalog_csv = _resolve_path(str(catalog_cfg.catalog_csv))
    logger.info("Loading sampled catalog from {}", catalog_csv)
    df = pd.read_csv(catalog_csv, dtype={"product_id": str})
    
    # Ensure all required curation columns are present
    df["im_url"] = df["im_url"].astype(str).str.strip()
    
    def _pick_first_available(columns: List[str]) -> pd.Series:
        for col in columns:
            if col in df.columns:
                series = df[col]
                if series.dropna().astype(str).str.strip().ne("").any():
                    return series
        return pd.Series([""] * len(df), index=df.index)

    # Different catalogs use different product-name/description column names.
    df["im_name"] = _pick_first_available(["title", "im_name", "name", "product_name", "product_title"])
    df["im_description"] = _pick_first_available([
        "description",
        "im_description",
        "product_description",
        "short_description",
    ])

    df["im_name"] = df["im_name"].fillna("")
    df["im_description"] = df["im_description"].fillna("")
    return df


def _resolve_per_gpu(cfg: DictConfig) -> int:
    devices = [str(x) for x in cfg.inference.devices]
    workers = int(cfg.inference.get("workers", 0))
    if workers > 0:
        return max(1, int(math.ceil(workers / len(devices))))
    return int(cfg.inference.per_gpu)


def _run_detection(sampled_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    per_gpu = _resolve_per_gpu(cfg)

    detector = ParallelDetect(
        model_dir=str(_resolve_path(str(cfg.model.model_dir))),
        detector="product",
        devices=[str(x) for x in cfg.inference.devices],
        per_gpu=per_gpu,
        category_overlap_thresh=float(cfg.inference.category_overlap_thresh),
        detector_kwargs={"score_thr": float(cfg.inference.score_thr)},
    )

    return detector.detect_dataframe(
        sampled_df,
        image_col="im_url",
        box_concept="all",
        multiple_object=True,
        detection_limit=int(cfg.inference.detection_limit),
    )


def _normalize_box(box: Iterable[Any]) -> List[int]:
    vals = list(box)
    if len(vals) != 4:
        raise ValueError(f"Invalid box length: {vals}")
    return [int(round(float(v))) for v in vals]


def _explode_detections(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for row in df.itertuples(index=False):
        dets_raw = getattr(row, "detections", None)
        if dets_raw is None:
            continue

        if isinstance(dets_raw, list):
            dets_iter = dets_raw
        elif isinstance(dets_raw, tuple):
            dets_iter = list(dets_raw)
        elif hasattr(dets_raw, "tolist"):
            dets_iter = dets_raw.tolist()
            if not isinstance(dets_iter, list):
                dets_iter = [dets_iter]
        else:
            dets_iter = [dets_raw]

        for det in dets_iter:
            if not isinstance(det, dict):
                continue
            if "name" not in det or "score" not in det or "box" not in det:
                continue
            try:
                det_box = _normalize_box(det["box"])
                det_score = float(det["score"])
            except Exception:
                continue
            rows.append(
                {
                    "catalog_name": getattr(row, "catalog_name"),
                    "product_id": getattr(row, "product_id"),
                    "category": getattr(row, "category"),
                    "im_url": getattr(row, "im_url"),
                    "im_id": getattr(row, "im_id"),
                    "im_name": getattr(row, "im_name"),
                    "im_description": getattr(row, "im_description"),
                    "det_name": str(det["name"]),
                    "det_score": det_score,
                    "det_box": det_box,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "catalog_name",
                "product_id",
                "category",
                "im_url",
                "im_id",
                "im_name",
                "im_description",
                "det_name",
                "det_score",
                "det_box",
            ]
        )

    return pd.DataFrame(rows)


def _cluster_indices(boxes: List[List[int]], iou_thresh: float) -> List[List[int]]:
    n = len(boxes)
    visited = [False] * n
    clusters: List[List[int]] = []

    adjacency: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if compute_iou(boxes[i], boxes[j]) >= iou_thresh:
                adjacency[i].append(j)
                adjacency[j].append(i)

    for i in range(n):
        if visited[i]:
            continue
        queue = [i]
        visited[i] = True
        comp: List[int] = []
        while queue:
            cur = queue.pop()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if not visited[nxt]:
                    visited[nxt] = True
                    queue.append(nxt)
        clusters.append(comp)

    return clusters


def _build_cluster_df(det_df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    iou_thresh = float(cfg.clustering.iou_cluster_thresh)
    out_rows: List[Dict[str, Any]] = []

    for _, group in det_df.groupby("im_url", sort=False):
        boxes = group["det_box"].tolist()
        indices = group.index.tolist()
        clusters = _cluster_indices(boxes, iou_thresh)

        for cluster_id, comp in enumerate(clusters):
            cluster_rows = group.loc[[indices[i] for i in comp]].copy()
            cluster_rows = cluster_rows.sort_values("det_score", ascending=False)
            top = cluster_rows.iloc[0]
            second_score = float(cluster_rows.iloc[1]["det_score"]) if len(cluster_rows) > 1 else 0.0
            margin = float(top["det_score"]) - second_score if len(cluster_rows) > 1 else float(top["det_score"])
            uncertainty = 1.0 - margin

            out_rows.append(
                {
                    "catalog_name": str(top["catalog_name"]),
                    "product_id": str(top["product_id"]),
                    "category": str(top["category"]),
                    "im_url": str(top["im_url"]),
                    "im_id": str(top["im_id"]),
                    "im_name": str(top["im_name"]),
                    "im_description": str(top["im_description"]),
                    "cluster_id": cluster_id,
                    "cluster_size": len(cluster_rows),
                    "top_class": str(top["det_name"]),
                    "top_score": float(top["det_score"]),
                    "second_score": second_score,
                    "margin": margin,
                    "uncertainty": uncertainty,
                    "top_box": top["det_box"],
                }
            )

    if not out_rows:
        return pd.DataFrame(
            columns=[
                "catalog_name",
                "product_id",
                "category",
                "im_url",
                "im_id",
                "im_name",
                "im_description",
                "cluster_id",
                "cluster_size",
                "top_class",
                "top_score",
                "second_score",
                "margin",
                "uncertainty",
                "top_box",
            ]
        )

    return pd.DataFrame(out_rows)


def _issue_rows_from_annotations_and_clusters(
    annotation_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    relevant_classes: set[str],
    cfg: DictConfig,
) -> pd.DataFrame:
    """Classifies issues into simple 'recall' and 'precision' categories without per-class thresholds."""
    model_score_threshold = float(cfg.run.get("model_score_threshold", 0.3))

    annotation_cols = annotation_df[["im_id", "visible_classes", "gpt_reason"]].copy()
    
    # 1. Filter model detections by score threshold
    if not cluster_df.empty:
        filtered_cluster_df = cluster_df[cluster_df["top_score"] >= model_score_threshold].copy()
    else:
        filtered_cluster_df = pd.DataFrame(columns=cluster_df.columns)

    if filtered_cluster_df.empty:
        merged_df = pd.DataFrame(columns=list(cluster_df.columns) + ["visible_classes", "visible_class_set", "gpt_reason"])
    else:
        merged_df = filtered_cluster_df.merge(annotation_cols, on="im_id", how="left")
        merged_df["visible_classes"] = merged_df["visible_classes"].fillna("[]").apply(_parse_class_list)
        merged_df["visible_class_set"] = merged_df["visible_classes"].apply(set)
        merged_df["top_class"] = merged_df["top_class"].astype(str).str.lower()
        merged_df["gpt_reason"] = merged_df["gpt_reason"].fillna("")

    # Classify precision issues and non-issues
    precision_rows: List[pd.Series] = []
    no_issue_rows: List[pd.Series] = []

    for row in merged_df.itertuples(index=False):
        top_class = str(row.top_class)
        if top_class in relevant_classes:
            if top_class in getattr(row, "visible_class_set", set()):
                # Detected and LLM confirmed
                row_dict = row._asdict()
                row_dict["box_status"] = "no_issue"
                row_dict["box_source"] = "model"
                row_dict["issue_type"] = "no_issue"
                no_issue_rows.append(pd.Series(row_dict))
            else:
                # Detected but LLM did not confirm (Precision issue / false positive)
                row_dict = row._asdict()
                row_dict["box_status"] = "fp_incorrect"
                row_dict["box_source"] = "model"
                row_dict["issue_type"] = "precision"
                precision_rows.append(pd.Series(row_dict))
        else:
            # Non-relevant classes (e.g. top, bottom, shoe, outerwear) are kept as no_issue
            row_dict = row._asdict()
            row_dict["box_status"] = "no_issue"
            row_dict["box_source"] = "model"
            row_dict["issue_type"] = "no_issue"
            no_issue_rows.append(pd.Series(row_dict))

    precision_df = pd.DataFrame(precision_rows)
    no_issue_df = pd.DataFrame(no_issue_rows)

    # 2. Classify recall issues (False Negatives)
    annotation_df_parsed = annotation_df.copy()
    annotation_df_parsed["visible_classes"] = annotation_df_parsed["visible_classes"].apply(_parse_class_list)
    
    # Gather what was detected per image
    detected_lookup = merged_df.groupby("im_id")["top_class"].apply(
        lambda s: set([str(x).strip().lower() for x in s.tolist()])
    )

    type2_rows: List[Dict[str, Any]] = []
    for anno_row in annotation_df_parsed.itertuples(index=False):
        visible_relevant = [
            cls for cls in getattr(anno_row, "visible_classes", [])
            if cls in relevant_classes
        ]
        detected_classes = (
            detected_lookup.get(anno_row.im_id, set())
            if anno_row.im_id in detected_lookup.index
            else set()
        )
        missing_classes = [cls for cls in visible_relevant if cls not in detected_classes]
        if not missing_classes:
            continue

        for class_name in missing_classes:
            type2_rows.append(
                {
                    "catalog_name": anno_row.catalog_name,
                    "product_id": anno_row.product_id,
                    "category": anno_row.category,
                    "im_url": anno_row.im_url,
                    "im_id": anno_row.im_id,
                    "im_name": anno_row.im_name,
                    "im_description": anno_row.im_description,
                    "cluster_id": -1,
                    "cluster_size": 0,
                    "top_class": class_name,
                    "top_score": 0.0,
                    "second_score": 0.0,
                    "margin": 0.0,
                    "uncertainty": 1.0,
                    "top_box": [0, 0, 0, 0],
                    "visible_classes": getattr(anno_row, "visible_classes", []),
                    "visible_class_set": set(),
                    "gpt_reason": getattr(anno_row, "gpt_reason", ""),
                    "box_status": "type2_missing",
                    "box_source": "model_missed",
                    "issue_type": "recall",
                }
            )

    type2_df = pd.DataFrame(type2_rows)

    parts: List[pd.DataFrame] = [precision_df, no_issue_df]
    if not type2_df.empty:
        parts.append(type2_df)
    
    # Filter empty DataFrames out
    parts = [p for p in parts if not p.empty]
    
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=list(cluster_df.columns) + ["box_status", "box_source", "issue_type"])


def _to_image_level_stl_rows(
    cluster_df: pd.DataFrame, optional_value: str, default_status: str, default_source: str
) -> pd.DataFrame:
    if cluster_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    meta: Dict[str, Dict[str, str]] = {}

    for row in cluster_df.itertuples(index=False):
        grouped[row.im_url].append(
            {
                "name": row.top_class,
                "box": row.top_box,
                "status": getattr(row, "box_status", default_status) or default_status,
                "source": getattr(row, "box_source", default_source) or default_source,
                "reason": str(getattr(row, "gpt_reason", "") or ""),
                "issue_type": str(getattr(row, "issue_type", "no_issue") or "no_issue"),
            }
        )
        meta[row.im_url] = {
            "im_id": row.im_id,
            "im_name": row.im_name,
            "im_description": row.im_description,
        }

    rows: List[Dict[str, Any]] = []
    for im_url, dets in grouped.items():
        concepts = ",".join([d["name"] for d in dets])
        boxes = ",".join([f"[{b[0]},{b[1]},{b[2]},{b[3]}]" for b in [d["box"] for d in dets]])
        statuses = ",".join([str(d["status"]) for d in dets])
        sources = ",".join([str(d["source"]) for d in dets])
        reasons = json.dumps([str(d["reason"]) for d in dets], ensure_ascii=True)
        issue_types = ",".join([str(d["issue_type"]) for d in dets])
        optional = ",".join([optional_value] * len(dets)) if len(dets) > 1 else optional_value

        rows.append(
            {
                "original_url": im_url,
                "im_url": im_url,
                "im_id": meta[im_url]["im_id"],
                "crop_box": "",
                "crop_box_concept": "",
                "im_name": meta[im_url]["im_name"],
                "im_description": meta[im_url]["im_description"],
                "labels": "",
                "concepts": concepts,
                "boxes": boxes,
                "optional": optional,
                "box_statuses": statuses,
                "box_sources": sources,
                "box_reasons": reasons,
                "issue_types": issue_types,
            }
        )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def _write_csv(df: pd.DataFrame, path_obj: Path) -> None:
    path_obj.parent.makedirs_p()
    df.to_csv(path_obj, index=False)
    logger.info("Wrote {} rows to {}", len(df), path_obj)


@app.command()
def main(
    config_file: str = typer.Argument(..., help="Path to curation V3 config YAML"),
):
    cfg = OmegaConf.load(str(config_file))

    annotation_df = _load_and_sample_catalogs(cfg)
    if annotation_df.empty:
        logger.warning("No sampled rows found. Exiting.")
        return

    output_dir = _resolve_path(str(cfg.run.output_dir))
    output_dir.makedirs_p()

    relevant_classes = set([str(x).strip().lower() for x in cfg.classes.relevant_classes])

    raw_det_path = output_dir / "detections_raw.parquet"
    if raw_det_path.exists():
        logger.info("Found existing raw detections at {}. Loading...", raw_det_path)
        det_full_df = pd.read_parquet(raw_det_path)
    else:
        logger.info("Running detection on {} images using model key {}", len(annotation_df), cfg.model.key)
        det_full_df = _run_detection(annotation_df, cfg)
        if bool(cfg.run.save_intermediate):
            det_full_df.to_parquet(raw_det_path, index=False)

    det_df = _explode_detections(det_full_df)
    cluster_df = _build_cluster_df(det_df, cfg)

    if bool(cfg.run.save_intermediate):
        det_df.to_parquet(output_dir / "detections_exploded.parquet", index=False)
        cluster_df.to_parquet(output_dir / "clusters.parquet", index=False)

    logger.info("Running issue classification rules...")
    issue_df = _issue_rows_from_annotations_and_clusters(
        annotation_df=annotation_df,
        cluster_df=cluster_df,
        relevant_classes=relevant_classes,
        cfg=cfg,
    )

    curated_out = _to_image_level_stl_rows(issue_df, optional_value="true", default_status="no_issue", default_source="model")
    curated_path_cfg = str(getattr(cfg.output, "curated_csv", "") or "").strip()
    curated_path = _resolve_path(curated_path_cfg) if curated_path_cfg else (output_dir / "curated_issues.csv")
    _write_csv(curated_out, curated_path)

    logger.info("Curation v3 completed. rows={}", len(curated_out))


if __name__ == "__main__":
    app()
