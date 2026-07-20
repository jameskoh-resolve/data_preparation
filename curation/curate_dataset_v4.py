#!/usr/bin/env python
"""Curate dataset V4 script with multiplicity LLM annotation and Locate Anything endpoint resolution.

Identifies incorrect boxes, finds recall issues using multiplicity constraints,
and queries the Locate Anything API to recover missing classes.
"""

from __future__ import annotations

import ast
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd
import requests
import typer
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from path import Path
from pydantic import BaseModel, Field
from tqdm import tqdm

# Load environment credentials from ~/.dltk.config or .env
_dltk_config = Path.home() / ".dltk.config"
if _dltk_config.exists():
    load_dotenv(_dltk_config)
load_dotenv()

app = typer.Typer(pretty_exceptions_show_locals=False)

REPO_ROOT = Path(os.path.abspath(__file__)).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detection.geometry import compute_iou
from detection.parallel_detect import ParallelDetect
from utils.vis_image import ImageContentCache
from llm.executor import LLMExecutor
from llm.gpt_anno_helpers import BaseAnnotator
from detection.dataset_utils import download_image

IDENTIFY_CLASSES_SYSTEM_PATH = REPO_ROOT / "llm" / "prompts" / "curate_dataset_one_pass_system.txt"
IDENTIFY_CLASSES_PROMPT_PATH = REPO_ROOT / "llm" / "prompts" / "identify_classes_prompt_v4.txt"
    
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


class IdentifyClassesResult(BaseModel):
    visible_classes: List[str] = Field(default_factory=list, description="Target classes visible in the image")
    gpt_reason: str = Field(..., description="Brief reasoning for the decision")


class IdentifyClassesAnnotator(BaseAnnotator):
    annotation_output_fields = ["visible_classes", "gpt_reason"]
    annotation_output_object = IdentifyClassesResult

    def __init__(self, prompt_path: Path, model_name: str = "gpt-4.1-mini-2025-04-14", prod_type: str = "fashion"):
        self.prompt_path = prompt_path
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


def _parse_visible_classes_multiplicity(visible_classes_json: str) -> Dict[str, str]:
    """Parse JSON visible_classes and determine base class names and expected multiplicity."""
    raw_list = _parse_class_list(visible_classes_json)
    res = {}
    for item in raw_list:
        clean = str(item).strip().lower()
        if not clean:
            continue
        if "(multiple)" in clean:
            base_cls = clean.replace("(multiple)", "").strip()
            res[base_cls] = "multiple"
        else:
            res[clean] = "singular"
    return res


def _parse_class_list_v4(visible_classes_json: Any) -> List[str]:
    """Return clean base class names (without '(multiple)') from the multiplicity JSON."""
    parsed_map = _parse_visible_classes_multiplicity(str(visible_classes_json))
    return list(parsed_map.keys())


def _load_and_sample_catalogs(cfg: DictConfig) -> pd.DataFrame:
    catalog_cfg = cfg.catalogs[0]
    catalog_csv = _resolve_path(str(catalog_cfg.catalog_csv))
    logger.info("Loading catalog from {}", catalog_csv)
    df = pd.read_csv(catalog_csv, dtype={"product_id": str})
    
    df["im_url"] = df["im_url"].astype(str).str.strip()
    
    def _pick_first_available(columns: List[str]) -> pd.Series:
        for col in columns:
            if col in df.columns:
                series = df[col]
                if series.dropna().astype(str).str.strip().ne("").any():
                    return series
        return pd.Series([""] * len(df), index=df.index)

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

    detector_kwargs = {"score_thr": float(cfg.inference.score_thr)}
    if "threshold_file" in cfg.model:
        detector_kwargs["threshold_file"] = str(cfg.model.threshold_file)

    model_dir_path = _resolve_path(str(cfg.model.model_dir))
    network_info_path = Path(model_dir_path) / "weardex_network_info.txt"
    detector_type = "centernet"
    if network_info_path.exists():
        try:
            net_type = network_info_path.read_text().strip().lower()
            if net_type == "rtmdet":
                detector_type = "rtmdet"
        except Exception:
            pass

    detector = ParallelDetect(
        model_dir=str(model_dir_path),
        detector=detector_type,
        devices=[str(x) for x in cfg.inference.devices],
        per_gpu=per_gpu,
        category_overlap_thresh=float(cfg.inference.category_overlap_thresh),
        detector_kwargs=detector_kwargs,
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


def _annotate_classes(
    sampled_df: pd.DataFrame,
    cfg: DictConfig,
    output_dir: Path,
    available_classes: str,
) -> pd.DataFrame:
    """Run LLM annotation to identify visible target classes (with multiplicity)."""
    if sampled_df.empty:
        return sampled_df

    llm_cfg = cfg.get("llm", {})
    model_name = str(llm_cfg.get("model", "gpt-4.1-mini-2025-04-14"))
    workers = int(llm_cfg.get("workers", 4))
    checkpoint_root = _resolve_path(str(llm_cfg.get("checkpoint_dir", str(output_dir / "llm_anno"))))
    checkpoint_dir = checkpoint_root / "identify_classes"

    from llm.gpt_anno_helpers import annotate_df as llm_annotate_df

    anno_rows = sampled_df.copy()
    anno_rows["available_classes"] = available_classes
    anno_rows["llm_image_source"] = anno_rows["im_url"].astype(str)

    prompt_path = _resolve_path(str(llm_cfg.get("prompt_path", "llm/prompts/identify_classes_prompt_v4.txt")))
    annotator = IdentifyClassesAnnotator(prompt_path=prompt_path, model_name=model_name, prod_type="fashion")
    
    annotated = llm_annotate_df(
        df=anno_rows,
        annotation_keys=["im_id"],
        annotator=annotator,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=10,
        max_workers=workers,
    )

    anno_keep_cols = ["im_id", "visible_classes", "gpt_reason"]
    annotated = annotated[anno_keep_cols].drop_duplicates(subset=["im_id"], keep="last")
    annotated = sampled_df.merge(annotated, on="im_id", how="left")

    annotated["visible_classes"] = annotated["visible_classes"].fillna("[]")
    annotated["gpt_reason"] = annotated["gpt_reason"].fillna("annotation_missing")

    return annotated


def _write_annotation_outputs(annotated_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.makedirs_p()
    annotated_df.to_csv(output_dir / "class_annotations.csv", index=False)

    visible_counter: Dict[str, int] = defaultdict(int)
    for classes in annotated_df.get("visible_classes", []):
        for class_name in _parse_class_list_v4(classes):
            visible_counter[class_name] += 1

    stats = {
        "total_images": int(len(annotated_df)),
        "visible_class_counts": dict(sorted(visible_counter.items())),
    }
    (output_dir / "class_annotation_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


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


def get_box_area(box: List[int]) -> float:
    """Calculate area of a bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def compute_io_min(box_a, box_b) -> float:
    """Compute Intersection over Minimum area (ioMin) between two boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = get_box_area(box_a)
    area_b = get_box_area(box_b)
    min_area = min(area_a, area_b)
    return inter / min_area if min_area > 0 else 0.0


def _run_locate_anything_single(
    item: dict,
    locate_anything_cfg: DictConfig,
) -> List[Dict[str, Any]]:
    endpoint_url = str(locate_anything_cfg.get("endpoint_url", "http://localhost:8080/detect_classes"))
    if endpoint_url.endswith("/detect"):
        endpoint_url = endpoint_url.replace("/detect", "/detect_classes")
        
    max_classes_per_prompt = int(locate_anything_cfg.get("max_classes_per_prompt", 4))
    decoding_mode = str(locate_anything_cfg.get("decoding_mode", "hybrid"))
    
    image_path = item["image_path"]
    recall_classes = item["recall_classes"]
    
    class_to_prompt = {}
    for rc in recall_classes:
        prompt_name = rc.replace("_", " ")
        class_to_prompt[prompt_name] = rc
    target_classes = sorted(class_to_prompt.keys())
    
    api_dets = []
    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            data = {
                "classes": json.dumps(target_classes),
                "max_classes_per_prompt": max_classes_per_prompt,
                "decoding_mode": decoding_mode,
            }
            response = requests.post(endpoint_url, files=files, data=data, timeout=30)
            response.raise_for_status()
            
        result = response.json()
        recall_set = set(recall_classes)
        for det in result.get("detections", []):
            label = str(det.get("label", "")).strip().lower()
            bbox = det.get("bbox")
            score = float(det.get("score", 1.0))
            if not label or not bbox:
                continue
                
            mapped_label = class_to_prompt.get(label, label)
            if mapped_label not in class_to_prompt:
                for pk, vk in class_to_prompt.items():
                    if pk.lower() == label.lower():
                        mapped_label = vk
                        break
                    if pk.replace(" ", "_").lower() == label.replace(" ", "_").lower():
                        mapped_label = vk
                        break
                        
            if mapped_label in recall_set:
                api_dets.append({
                    "name": mapped_label,
                    "box": bbox,
                    "score": score,
                    "source": "locate_anything",
                })
    except Exception as e:
        logger.error("Locate anything API call failed for {}: {}", item["im_url"], e)
    return api_dets


def _run_locate_anything_batch(
    images_to_query: List[dict],
    locate_anything_cfg: DictConfig,
) -> List[List[Dict[str, Any]]]:
    endpoint_url = str(locate_anything_cfg.get("endpoint_url", "http://localhost:8080/detect_classes_batch"))
    if endpoint_url.endswith("/detect_classes"):
        endpoint_url += "_batch"
    elif endpoint_url.endswith("/detect"):
        endpoint_url = endpoint_url.replace("/detect", "/detect_classes_batch")
        
    batch_size = int(locate_anything_cfg.get("batch_size", 4))
    max_classes_per_prompt = int(locate_anything_cfg.get("max_classes_per_prompt", 4))
    decoding_mode = str(locate_anything_cfg.get("decoding_mode", "hybrid"))
    
    batches = [images_to_query[i : i + batch_size] for i in range(0, len(images_to_query), batch_size)]
    all_results: List[List[Dict[str, Any]]] = []
    
    for batch in batches:
        opened_files = []
        try:
            files = []
            data = [
                ("max_classes_per_prompt", str(max_classes_per_prompt)),
                ("decoding_mode", str(decoding_mode)),
            ]
            batch_class_maps = []
            for item in batch:
                image_path = item["image_path"]
                f = open(image_path, "rb")
                opened_files.append(f)
                files.append(("images", (os.path.basename(image_path), f, "image/jpeg")))
                
                class_to_prompt = {}
                for rc in item["recall_classes"]:
                    prompt_name = rc.replace("_", " ")
                    class_to_prompt[prompt_name] = rc
                batch_class_maps.append(class_to_prompt)
                
                target_classes = sorted(class_to_prompt.keys())
                data.append(("classes", json.dumps(target_classes)))
                
            response = requests.post(endpoint_url, files=files, data=data, timeout=60)
            response.raise_for_status()
            
            res_json = response.json()
            if isinstance(res_json, dict) and "results" in res_json:
                results = res_json["results"]
            elif isinstance(res_json, list):
                results = res_json
            else:
                results = []
                
            for idx, item in enumerate(batch):
                api_dets = []
                if idx < len(results):
                    det_res = results[idx]
                    detections = det_res.get("detections", []) if isinstance(det_res, dict) else []
                    class_to_prompt = batch_class_maps[idx]
                    recall_set = set(item["recall_classes"])
                    for det in detections:
                        label = str(det.get("label", "")).strip().lower()
                        bbox = det.get("bbox")
                        score = float(det.get("score", 1.0))
                        if not label or not bbox:
                            continue
                            
                        mapped_label = class_to_prompt.get(label, label)
                        if mapped_label not in class_to_prompt:
                            for pk, vk in class_to_prompt.items():
                                if pk.lower() == label.lower():
                                    mapped_label = vk
                                    break
                                if pk.replace(" ", "_").lower() == label.replace(" ", "_").lower():
                                    mapped_label = vk
                                    break
                                    
                        if mapped_label in recall_set:
                            api_dets.append({
                                "name": mapped_label,
                                "box": bbox,
                                "score": score,
                                "source": "locate_anything",
                            })
                all_results.append(api_dets)
        except Exception as e:
            logger.error("Locate anything batch API call failed: {}", e)
            for _ in batch:
                all_results.append([])
        finally:
            for f in opened_files:
                f.close()
    return all_results


def _suppress_duplicates(combined_dets: List[dict], locate_anything_cfg: DictConfig) -> List[dict]:
    by_class = defaultdict(list)
    for d in combined_dets:
        by_class[d["name"]].append(d)
        
    final_dets = []
    iou_thresh = float(locate_anything_cfg.get("iou_thresh", 0.7))
    io_min_thresh = float(locate_anything_cfg.get("io_min_thresh", 0.7))
    
    for cls_name, cls_list in by_class.items():
        cls_list = sorted(cls_list, key=lambda x: get_box_area(x["box"]))
        accepted = []
        for d in cls_list:
            box = d["box"]
            should_suppress = False
            for acc in accepted:
                iou = compute_iou(box, acc["box"])
                io_min = compute_io_min(box, acc["box"])
                if iou >= iou_thresh or io_min >= io_min_thresh:
                    should_suppress = True
                    break
            if not should_suppress:
                accepted.append(d)
        final_dets.extend(accepted)
    return final_dets


def _process_image_curation_v4(
    im_id: str,
    im_url: str,
    visible_classes_json: str,
    image_detections: pd.DataFrame,
    relevant_classes: Set[str],
    locate_anything_cfg: DictConfig,
    cache_dir: Path,
) -> List[Dict[str, Any]]:
    """Process a single image, removing incorrect boxes and resolving recall issues via API."""
    multiplicity_map = _parse_visible_classes_multiplicity(visible_classes_json)
    multiplicity_map = {k: v for k, v in multiplicity_map.items() if k in relevant_classes}
    
    correct_model_dets = []
    detected_counts = defaultdict(int)
    
    for row in image_detections.itertuples(index=False):
        top_class = str(row.top_class).strip().lower()
        if top_class in relevant_classes:
            if top_class in multiplicity_map:
                correct_model_dets.append({
                    "name": top_class,
                    "box": row.top_box,
                    "score": float(row.top_score),
                    "source": "model",
                })
                detected_counts[top_class] += 1
        else:
            correct_model_dets.append({
                "name": top_class,
                "box": row.top_box,
                "score": float(row.top_score),
                "source": "model",
            })
            detected_counts[top_class] += 1
            
    recall_classes = []
    for c, multiplicity in multiplicity_map.items():
        count = detected_counts[c]
        if multiplicity == "multiple" and count <= 1:
            recall_classes.append(c)
        elif multiplicity == "singular" and count == 0:
            recall_classes.append(c)
            
    api_dets = []
    if recall_classes:
        try:
            _, image_path = download_image(im_url, cache_dir)
            if image_path is not None and os.path.exists(image_path):
                item = {
                    "im_url": im_url,
                    "image_path": image_path,
                    "recall_classes": recall_classes,
                }
                api_dets = _run_locate_anything_single(item, locate_anything_cfg)
            else:
                logger.warning("Could not download image {} to query locate anything", im_url)
        except Exception as e:
            logger.error("Locate anything API call failed for {}: {}", im_url, e)
            
    combined_dets = correct_model_dets + api_dets
    return _suppress_duplicates(combined_dets, locate_anything_cfg)


@app.command()
def main(
    config_file: str = typer.Argument(..., help="Path to curation V4 config YAML"),
):
    cfg = OmegaConf.load(str(config_file))

    sampled_df = _load_and_sample_catalogs(cfg)
    if sampled_df.empty:
        logger.warning("No sampled rows found. Exiting.")
        return

    output_dir = _resolve_path(str(cfg.run.output_dir))
    output_dir.makedirs_p()

    relevant_classes = set([str(x).strip().lower() for x in cfg.classes.relevant_classes])
    configured_classes = str(cfg.get("llm", {}).get("available_classes", "") or "").strip()
    available_classes = configured_classes or ", ".join(sorted(relevant_classes))

    logger.info("Running LLM class identification on {} images...", len(sampled_df))
    annotation_df = _annotate_classes(
        sampled_df=sampled_df,
        cfg=cfg,
        output_dir=output_dir,
        available_classes=available_classes,
    )
    _write_annotation_outputs(annotation_df, output_dir)

    raw_det_path = output_dir / "detections_raw.parquet"
    if raw_det_path.exists():
        logger.info("Found existing raw detections at {}. Loading...", raw_det_path)
        det_full_df = pd.read_parquet(raw_det_path)
    else:
        logger.info("Running initial detection using model {}", cfg.model.key)
        det_full_df = _run_detection(sampled_df, cfg)
        if bool(cfg.run.save_intermediate):
            det_full_df.to_parquet(raw_det_path, index=False)

    det_df = _explode_detections(det_full_df)
    cluster_df = _build_cluster_df(det_df, cfg)

    if bool(cfg.run.save_intermediate):
        det_df.to_parquet(output_dir / "detections_exploded.parquet", index=False)
        cluster_df.to_parquet(output_dir / "clusters.parquet", index=False)

    logger.info("Resolving recall/precision issues with Locate Anything API...")
    locate_anything_cfg = cfg.get("locate_anything", {})
    cache_dir = output_dir / "_image_cache"
    cache_dir.makedirs_p()
    
    batch_processing = bool(locate_anything_cfg.get("batch_processing", False))
    
    items_to_process = []
    
    for row in annotation_df.itertuples(index=False):
        im_id = row.im_id
        im_url = row.im_url
        im_name = row.im_name
        im_description = row.im_description
        visible_classes_json = row.visible_classes
        gpt_reason = getattr(row, "gpt_reason", "")
        
        if not cluster_df.empty:
            image_detections = cluster_df[cluster_df["im_id"] == im_id]
        else:
            image_detections = pd.DataFrame(columns=cluster_df.columns)
            
        multiplicity_map = _parse_visible_classes_multiplicity(visible_classes_json)
        multiplicity_map = {k: v for k, v in multiplicity_map.items() if k in relevant_classes}
        
        correct_model_dets = []
        detected_counts = defaultdict(int)
        
        for det_row in image_detections.itertuples(index=False):
            top_class = str(det_row.top_class).strip().lower()
            if top_class in relevant_classes:
                if top_class in multiplicity_map:
                    correct_model_dets.append({
                        "name": top_class,
                        "box": det_row.top_box,
                        "score": float(det_row.top_score),
                        "source": "model",
                    })
                    detected_counts[top_class] += 1
            else:
                correct_model_dets.append({
                    "name": top_class,
                    "box": det_row.top_box,
                    "score": float(det_row.top_score),
                    "source": "model",
                })
                detected_counts[top_class] += 1
                
        recall_classes = []
        for c, multiplicity in multiplicity_map.items():
            count = detected_counts[c]
            if multiplicity == "multiple" and count <= 1:
                recall_classes.append(c)
            elif multiplicity == "singular" and count == 0:
                recall_classes.append(c)
                
        items_to_process.append({
            "im_id": im_id,
            "im_url": im_url,
            "im_name": im_name,
            "im_description": im_description,
            "gpt_reason": gpt_reason,
            "correct_model_dets": correct_model_dets,
            "recall_classes": recall_classes,
            "image_path": None,
            "api_dets": [],
        })

    images_to_query = []
    for item in items_to_process:
        if item["recall_classes"]:
            try:
                _, image_path = download_image(item["im_url"], cache_dir)
                if image_path is not None and os.path.exists(image_path):
                    item["image_path"] = image_path
                    images_to_query.append(item)
                else:
                    logger.warning("Could not download image {} to query locate anything", item["im_url"])
            except Exception as e:
                logger.error("Download failed for locate anything input: {}", e)

    if images_to_query:
        if batch_processing:
            logger.info("Querying locate anything API with batch processing (batch_size={}) for {} images", locate_anything_cfg.get("batch_size", 4), len(images_to_query))
            all_api_dets = _run_locate_anything_batch(images_to_query, locate_anything_cfg)
            for item, api_dets in zip(images_to_query, all_api_dets):
                item["api_dets"] = api_dets
        else:
            logger.info("Querying locate anything API sequentially for {} images", len(images_to_query))
            for item in tqdm(images_to_query, desc="Querying locate anything"):
                item["api_dets"] = _run_locate_anything_single(item, locate_anything_cfg)

    final_rows = []
    for item in items_to_process:
        combined_dets = item["correct_model_dets"] + item["api_dets"]
        final_dets = _suppress_duplicates(combined_dets, locate_anything_cfg)
        
        for det in final_dets:
            final_rows.append({
                "im_id": item["im_id"],
                "im_url": item["im_url"],
                "im_name": item["im_name"],
                "im_description": item["im_description"],
                "top_class": det["name"],
                "top_box": det["box"],
                "box_status": "no_issue",
                "box_source": det["source"],
                "gpt_reason": item["gpt_reason"],
                "issue_type": "no_issue",
            })
            
    if not final_rows:
        final_df = pd.DataFrame(columns=[
            "im_id", "im_url", "im_name", "im_description",
            "top_class", "top_box", "box_status", "box_source",
            "gpt_reason", "issue_type"
        ])
    else:
        final_df = pd.DataFrame(final_rows)

    curated_out = _to_image_level_stl_rows(final_df, optional_value="true", default_status="no_issue", default_source="model")
    curated_path_cfg = str(getattr(cfg.output, "curated_csv", "") or "").strip()
    curated_path = _resolve_path(curated_path_cfg) if curated_path_cfg else (output_dir / "curated_issues.csv")
    _write_csv(curated_out, curated_path)

    logger.info("Curation v4 completed. rows={}", len(curated_out))


if __name__ == "__main__":
    app()
