#!/usr/bin/env python
"""Auto-Annotate curation script.

Runs fashion_model (local RTMDet) and/or locate_anything detectors on datasets,
applies deduplication policies (biggest/smallest/locate_anything/fashion_model preference),
and validates candidate crops using LLM-based verification.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np
import pandas as pd
import requests
import typer
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detection.geometry import compute_iou
from detection.dataset_utils import download_image
from utils.vis_image import get_image_content
from llm.executor import LLMExecutor
from llm.utils import resolve_model_alias

app = typer.Typer(pretty_exceptions_show_locals=False)


class VerificationResult(BaseModel):
    is_valid: bool = Field(..., description="Whether the crop contains the specified class of accessory or item.")
    reason: str = Field(..., description="Short explanation of the validation result.")


class DetectorRegistry:
    _detector_instances = {}

    @classmethod
    def get_detector(cls, model_dir: str, threshold_file: str):
        key = (model_dir, threshold_file)
        if key not in cls._detector_instances:
            model_dir_path = Path(model_dir)
            network_info_path = model_dir_path / "weardex_network_info.txt"
            detector_type = "centernet"
            if network_info_path.exists():
                try:
                    net_type = network_info_path.read_text().strip().lower()
                    if net_type in ("rtmdet", "simple"):
                        detector_type = net_type
                except Exception:
                    pass

            logger.info("Loading {} detector from {} with threshold file {}", detector_type, model_dir, threshold_file)
            if detector_type == "rtmdet":
                from detection.rtmdet_detect import RTMDetDetect
                detector = RTMDetDetect(model_dir=str(model_dir_path), threshold_file=threshold_file)
            elif detector_type == "simple":
                from detection.simple_detect import SimpleDetect
                detector = SimpleDetect(model_dir=str(model_dir_path), threshold_file=threshold_file)
            else:
                from detection.centernet_detect import CenterNetDetect
                detector = CenterNetDetect(model_dir=str(model_dir_path), threshold_file=threshold_file)

            cls._detector_instances[key] = detector
        return cls._detector_instances[key]


def _stable_image_id(im_url: str) -> str:
    return hashlib.md5(str(im_url).encode("utf-8")).hexdigest()[:16]


def safe_format(template: str, **kwargs) -> str:
    for k, v in kwargs.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def parse_existing_boxes_and_concepts(row: pd.Series) -> List[Dict[str, Any]]:
    raw_boxes = row.get("boxes")
    raw_concepts = row.get("concepts")

    if pd.isna(raw_boxes) or pd.isna(raw_concepts):
        return []

    raw_boxes = str(raw_boxes).strip()
    raw_concepts = str(raw_concepts).strip()

    if not raw_boxes or not raw_concepts:
        return []

    concepts = [c.strip() for c in raw_concepts.split(",") if c.strip()]
    boxes = []

    # Find pattern [x,y,x,y]
    matches = re.findall(r"\[([^\]]+)\]", raw_boxes)
    if matches:
        for match in matches:
            try:
                box = [float(v) for v in match.split(",")]
                if len(box) == 4:
                    boxes.append(box)
            except Exception:
                pass
    else:
        # Fallback to comma-separated single box
        try:
            box = [float(v) for v in raw_boxes.split(",")]
            if len(box) == 4:
                boxes.append(box)
        except Exception:
            pass

    results = []
    for i in range(min(len(concepts), len(boxes))):
        results.append({
            "name": concepts[i],
            "box": boxes[i],
            "score": 1.0,
            "source": "original"
        })
    return results


def compute_io_min(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    min_area = min(area_a, area_b)
    return inter / min_area if min_area > 0 else 0.0


def get_sort_key_fn(keep_which: str):
    keep_which = str(keep_which or "").strip().lower()

    def sort_key(d: dict):
        source = str(d.get("source", "")).strip().lower()
        box = d.get("box", [0, 0, 0, 0])
        score = float(d.get("score", 1.0))
        area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))

        if keep_which == "keep locate anything box":
            prim = 0 if source == "locate_anything" else 1
        elif keep_which == "keep fashion model box":
            prim = 0 if source == "fashion_model" else 1
        elif keep_which == "keep biggest":
            prim = -area
        elif keep_which == "keep smallest":
            prim = area
        else:
            prim = 0

        return (prim, -score)

    return sort_key


def suppress_duplicates(
    detections: List[Dict[str, Any]],
    iou_thresh: float,
    io_min_thresh: float,
    keep_which: str
) -> List[Dict[str, Any]]:
    by_class = defaultdict(list)
    for d in detections:
        by_class[d["name"]].append(d)

    final_dets = []
    sort_fn = get_sort_key_fn(keep_which)

    for cls_name, cls_list in by_class.items():
        sorted_list = sorted(cls_list, key=sort_fn)
        accepted = []
        for d in sorted_list:
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


def prepare_crop(
    image: np.ndarray,
    box: List[float],
    padding: float = 0.0,
    contrast: Any = None,
    brightness: Any = None
) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box

    box_w = x2 - x1
    box_h = y2 - y1

    if padding > 0:
        px = box_w * padding
        py = box_h * padding
        x1_pad = max(0.0, x1 - px)
        y1_pad = max(0.0, y1 - py)
        x2_pad = min(float(w), x2 + px)
        y2_pad = min(float(h), y2 + py)
    else:
        x1_pad, y1_pad, x2_pad, y2_pad = x1, y1, x2, y2

    x1_idx = max(0, int(round(x1_pad)))
    y1_idx = max(0, int(round(y1_pad)))
    x2_idx = min(w, int(round(x2_pad)))
    y2_idx = min(h, int(round(y2_pad)))

    crop = image[y1_idx:y2_idx, x1_idx:x2_idx]
    if crop.size == 0:
        crop = image[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))]

    if crop.size == 0:
        return crop

    # Contrast adjustment
    if contrast is not None:
        if isinstance(contrast, bool):
            if contrast:
                ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                crop = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        elif isinstance(contrast, (int, float)):
            crop = cv2.convertScaleAbs(crop, alpha=float(contrast), beta=0)
        elif isinstance(contrast, str):
            c_lower = contrast.lower()
            if c_lower in ("equalize", "histeq", "true"):
                ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                crop = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            elif c_lower == "clahe":
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
                crop = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Brightness adjustment
    if brightness is not None:
        try:
            beta_val = int(brightness)
            crop = cv2.convertScaleAbs(crop, alpha=1.0, beta=beta_val)
        except Exception:
            pass

    return crop


def _query_locate_anything_api(
    image_path: str,
    group_classes: List[str],
    endpoint_url: str,
    decoding_mode: str,
) -> List[Dict[str, Any]]:
    if endpoint_url.endswith("/detect"):
        endpoint_url = endpoint_url.replace("/detect", "/detect_classes")

    class_to_prompt = {rc.replace("_", " "): rc for rc in group_classes}
    target_classes = sorted(class_to_prompt.keys())

    api_dets = []
    try:
        with open(image_path, "rb") as f:
            files = {"image": (os.path.basename(image_path), f, "image/jpeg")}
            data = {
                "classes": json.dumps(target_classes),
                "max_classes_per_prompt": len(target_classes),
                "decoding_mode": decoding_mode,
            }
            logger.debug("Querying locate_anything at {} with classes={}", endpoint_url, target_classes)
            response = requests.post(endpoint_url, files=files, data=data, timeout=60)
            response.raise_for_status()

        result = response.json()
        group_set = set(group_classes)
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

            if mapped_label in group_set:
                api_dets.append({
                    "name": mapped_label,
                    "box": bbox,
                    "score": score,
                    "source": "locate_anything",
                })
    except Exception as e:
        logger.error("Locate anything API call failed for {}: {}", image_path, e)
    return api_dets


def run_locate_anything(
    image_path: str,
    model_cfg: dict,
) -> List[Dict[str, Any]]:
    endpoint_url = str(model_cfg.get("api_endpoint", model_cfg.get("endpoint_url", "http://localhost:8080/detect_classes")))
    decoding_mode = str(model_cfg.get("decoding_mode", "slow"))

    classes = model_cfg.get("classes", [])
    if not classes:
        logger.warning("No classes specified for locate_anything detector. Returning empty.")
        return []

    class_groups = model_cfg.get("class_groups")
    max_classes_per_prompt = model_cfg.get("max_classes_per_prompt")

    # Partition classes into groups
    groups = []
    if class_groups:
        groups = class_groups
    elif max_classes_per_prompt is not None:
        max_classes = int(max_classes_per_prompt)
        groups = [classes[i : i + max_classes] for i in range(0, len(classes), max_classes)]
    else:
        groups = [classes]

    all_dets = []
    for group in groups:
        group_dets = _query_locate_anything_api(image_path, group, endpoint_url, decoding_mode)
        all_dets.extend(group_dets)

    return all_dets


def run_fashion_model(image: np.ndarray, model_cfg: dict) -> List[Dict[str, Any]]:
    model_dir = model_cfg.get("model_dir")
    if not model_dir:
        raise ValueError("model_dir is required for fashion_model detector")

    threshold_file = model_cfg.get("threshold_file", "threshold.txt")
    detector = DetectorRegistry.get_detector(model_dir, threshold_file)

    results = detector.detect(image)

    # Convert results keys and construct detection list
    dets = []
    for r in results:
        dets.append({
            "name": r["name"],
            "box": r["box"],
            "score": r.get("score", 1.0),
            "source": "fashion_model"
        })

    # Filter by configured classes
    allowed_classes = [c.lower() for c in model_cfg.get("classes", [])]
    if allowed_classes:
        dets = [d for d in dets if d["name"].lower() in allowed_classes]

    return dets


def load_image_and_path(im_url: str, cache_dir: Path):
    if os.path.exists(im_url):
        image = cv2.imread(im_url)
        return image, Path(im_url)

    # Try downloading
    return download_image(im_url, cache_dir)


def validate_detection_with_llm(
    image: np.ndarray,
    det: dict,
    llm_cfg: dict,
    executor,
    system_prompt: str,
    user_prompt_tmpl: str,
) -> bool:
    class_name = det["name"]

    # Get class overrides
    class_override = llm_cfg.get(class_name, {})
    general_cfg = llm_cfg.get("general", {})

    def get_cfg_val(key, default=None):
        if key in class_override:
            return class_override[key]
        return general_cfg.get(key, default)

    padding = float(get_cfg_val("padding", 0.0))
    contrast = get_cfg_val("contrast", None)
    brightness = get_cfg_val("brightness", None)

    # Crop the box
    crop = prepare_crop(image, det["box"], padding=padding, contrast=contrast, brightness=brightness)
    if crop.size == 0:
        logger.warning("Empty crop for detection of class {}, skipping LLM validation", class_name)
        return True

    # Encode crop to JPEG bytes
    success, encoded_img = cv2.imencode(".jpg", crop)
    if not success:
        logger.warning("Failed to encode crop to JPEG for class {}, skipping LLM validation", class_name)
        return True
    crop_bytes = encoded_img.tobytes()

    # Determine class-specific task or prompt override
    custom_task = class_override.get("task", class_override.get("task_prompt", None))
    custom_prompt = class_override.get("prompt", None)

    if custom_task:
        task_str = safe_format(str(custom_task), class_name=class_name)
    else:
        task_str = f"Verify if the target accessory/clothing class '{class_name}' is clearly visible in this crop."

    if custom_prompt:
        prompt_val = str(custom_prompt)
        if os.path.exists(prompt_val):
            tmpl = Path(prompt_val).read_text().strip()
        else:
            tmpl = prompt_val
        user_prompt = safe_format(tmpl, class_name=class_name, task_prompt=task_str)
    else:
        user_prompt = safe_format(user_prompt_tmpl, class_name=class_name, task_prompt=task_str)

    from langchain.schema import SystemMessage, HumanMessage
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    try:
        parsed = executor.predict(
            messages,
            images=[crop_bytes],
            output_object_type=VerificationResult,
        )
        logger.info("LLM validation for {}: is_valid={}, reason={}", class_name, parsed.is_valid, parsed.reason)
        return parsed.is_valid
    except Exception as e:
        logger.error("LLM validation call failed for class {}: {}", class_name, e)
        return True


@app.command()
def main(
    config_file: str = typer.Argument(..., help="Path to auto-annotate config YAML"),
):
    """Run the auto-annotation curation pipeline."""
    cfg = OmegaConf.load(str(config_file))

    dataset_cfg = cfg.get("dataset", {})
    dsttype = dataset_cfg.get("type", "flat csv").lower()

    # 1. Retrieve dataset
    if dsttype == "hydravision":
        import data_factory.client.hydravision as hv
        dataset_name = dataset_cfg.get("dataset_name")
        if not dataset_name:
            raise ValueError("dataset_name is required for hydravision dataset type.")
        logger.info("Retrieving HydraVision dataset: {}", dataset_name)
        df = hv.HydraVisionGetDataset(dataset_name).read_dataframe()
    else:
        path = dataset_cfg.get("path")
        if not path:
            raise ValueError("path is required for flat csv dataset type.")
        logger.info("Reading flat CSV from {}", path)
        df = pd.read_csv(path)

    if df.empty:
        logger.warning("Input dataset is empty. Exiting.")
        return

    # Normalize image URL column
    image_col = None
    for col in ("im_url", "image_url", "url", "image", "original_url"):
        if col in df.columns:
            image_col = col
            break

    if not image_col:
        raise ValueError(f"Could not find image URL column in dataset. Available: {list(df.columns)}")

    if "im_url" not in df.columns:
        df["im_url"] = df[image_col]

    if "im_id" not in df.columns:
        df["im_id"] = df["im_url"].apply(_stable_image_id)

    # Initialize cache dir
    output_dir = Path(dataset_cfg.get("output_dir", REPO_ROOT / "curated_datasets/curation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    custom_cache = dataset_cfg.get("cache_dir")
    if custom_cache:
        cache_dir = Path(custom_cache)
    else:
        cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 2. Setup LLM if validation configured
    llm_cfg = cfg.get("llm_validation", cfg.get("llm validation"))
    llm_executor = None
    system_prompt = ""
    user_prompt_tmpl = ""
    llm_classes = []

    if llm_cfg:
        logger.info("Setting up LLM Validation...")
        model_type = resolve_model_alias(str(llm_cfg.get("model_type", "gpt-4.1-mini")))
        llm_executor = LLMExecutor.from_model_name(model_type)

        prompt_path = llm_cfg.get("prompt")
        if prompt_path and os.path.exists(prompt_path):
            user_prompt_tmpl = Path(prompt_path).read_text().strip()
        else:
            default_prompt = REPO_ROOT / "llm/prompts/verify_classes_prompt.txt"
            if default_prompt.exists():
                user_prompt_tmpl = default_prompt.read_text().strip()
            else:
                user_prompt_tmpl = "Verify if the target accessory/clothing class '{class_name}' is clearly visible in this crop."

        system_prompt_path = REPO_ROOT / "llm/prompts/verify_classes_system.txt"
        if system_prompt_path.exists():
            system_prompt = system_prompt_path.read_text().strip()
        else:
            system_prompt = "You are a visual validation assistant."

        llm_classes = [c.lower() for c in llm_cfg.get("classes", [])]

    # 3. Process each row
    detection_models = cfg.get("detection_models", cfg.get("detection models", []))
    if isinstance(detection_models, dict):
        # Support dict format as single model or dict of models
        if "model_type" in detection_models:
            detection_models = [detection_models]
        else:
            detection_models = list(detection_models.values())

    keep_bbs = bool(dataset_cfg.get("keep_bounding_boxes", False))
    dedup_cfg = cfg.get("dedup_policy", cfg.get("dedup policy", {}))

    annotated_rows = []
    logger.info("Processing {} images...", len(df))

    for idx, row in df.iterrows():
        im_url = row["im_url"]
        logger.info("Processing image [{}/{}]: {}", idx + 1, len(df), im_url)

        # Download/read image
        image, local_path = load_image_and_path(im_url, cache_dir)
        if image is None:
            logger.warning("Failed to load image from {}. Skipping.", im_url)
            continue

        # Start with existing boxes if configured
        row_dets = []
        if keep_bbs:
            row_dets.extend(parse_existing_boxes_and_concepts(row))

        # Run configured detectors
        for model_cfg in detection_models:
            model_type = model_cfg.get("model_type")
            if model_type == "fashion_model":
                try:
                    fashion_dets = run_fashion_model(image, model_cfg)
                    row_dets.extend(fashion_dets)
                except Exception as e:
                    logger.error("Fashion model execution failed: {}", e)
            elif model_type == "locate_anything":
                try:
                    locate_dets = run_locate_anything(str(local_path), model_cfg)
                    row_dets.extend(locate_dets)
                except Exception as e:
                    logger.error("Locate anything execution failed: {}", e)

        # 4. Apply deduplication policy
        if dedup_cfg and row_dets:
            # We want to resolve policy for each class
            # To do this, we can run NMS class by class or group all detections
            by_class = defaultdict(list)
            for d in row_dets:
                by_class[d["name"]].append(d)

            deduped_dets = []
            for cls_name, cls_list in by_class.items():
                class_policy = dedup_cfg.get(cls_name)
                if class_policy is None:
                    class_policy = dedup_cfg.get("general")

                if class_policy:
                    iou_thresh = float(class_policy.get("IOU", 0.7))
                    io_min_thresh = float(class_policy.get("IOMin", 0.7))
                    keep_which = str(class_policy.get("keep_which", "keep biggest"))
                    # Deduplicate this class
                    deduped_cls_list = suppress_duplicates(cls_list, iou_thresh, io_min_thresh, keep_which)
                    deduped_dets.extend(deduped_cls_list)
                else:
                    deduped_dets.extend(cls_list)

            row_dets = deduped_dets

        # 5. Apply LLM validation
        if llm_executor and row_dets:
            validated_dets = []
            for d in row_dets:
                # Check if we should validate this class
                should_validate = (not llm_classes) or (d["name"].lower() in llm_classes)
                if should_validate:
                    is_valid = validate_detection_with_llm(
                        image, d, llm_cfg, llm_executor, system_prompt, user_prompt_tmpl
                    )
                    if is_valid:
                        validated_dets.append(d)
                else:
                    validated_dets.append(d)
            row_dets = validated_dets

        # Form final row values
        concepts_str = ",".join([d["name"] for d in row_dets])
        boxes_str = ",".join([f"[{int(round(b[0]))},{int(round(b[1]))},{int(round(b[2]))},{int(round(b[3]))}]" for b in [d["box"] for d in row_dets]])

        row_updated = dict(row)
        row_updated["concepts"] = concepts_str
        row_updated["boxes"] = boxes_str
        annotated_rows.append(row_updated)

    # 6. Save results
    out_df = pd.DataFrame(annotated_rows)
    output_path = output_dir / "annotated_dataset.csv"
    out_df.to_csv(output_path, index=False)
    logger.info("Auto-annotation pipeline finished. Saved {} annotated rows to {}", len(out_df), output_path)


@app.command(name="prefetch")
def prefetch_cache(
    config_file: str = typer.Argument(..., help="Path to auto-annotate config YAML"),
):
    """Pre-download all images in the dataset into the local cache folder.

    Run this command on a machine with VIS/internet access before transferring
    the repo / cache directory to an isolated GPU server.
    """
    cfg = OmegaConf.load(str(config_file))
    dataset_cfg = cfg.get("dataset", {})
    dsttype = dataset_cfg.get("type", "flat csv").lower()

    if dsttype == "hydravision":
        import data_factory.client.hydravision as hv
        dataset_name = dataset_cfg.get("dataset_name")
        if not dataset_name:
            raise ValueError("dataset_name is required for hydravision dataset type.")
        logger.info("Retrieving HydraVision dataset: {}", dataset_name)
        df = hv.HydraVisionGetDataset(dataset_name).read_dataframe()
    else:
        path = dataset_cfg.get("path")
        if not path:
            raise ValueError("path is required for flat csv dataset type.")
        logger.info("Reading flat CSV from {}", path)
        df = pd.read_csv(path)

    image_col = None
    for col in ("im_url", "image_url", "url", "image", "original_url"):
        if col in df.columns:
            image_col = col
            break

    if not image_col:
        raise ValueError(f"Could not find image URL column in dataset. Available: {list(df.columns)}")

    output_dir = Path(dataset_cfg.get("output_dir", REPO_ROOT / "curated_datasets/curation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    custom_cache = dataset_cfg.get("cache_dir")
    if custom_cache:
        cache_dir = Path(custom_cache)
    else:
        cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    urls = df[image_col].dropna().unique()
    logger.info("Prefetching {} images into cache directory: {}", len(urls), cache_dir)

    success_count = 0
    for idx, url in enumerate(urls):
        url_str = str(url).strip()
        if not url_str:
            continue
        img, path_obj = load_image_and_path(url_str, cache_dir)
        if img is not None:
            success_count += 1
        if (idx + 1) % 50 == 0 or (idx + 1) == len(urls):
            logger.info("Downloaded/cached [{}/{}] images", idx + 1, len(urls))

    logger.info("Prefetch complete. {}/{} images successfully cached in {}", success_count, len(urls), cache_dir)


@app.command(name="prep-azure")
def prep_azure(
    config_file: str = typer.Argument(..., help="Path to auto-annotate config YAML"),
    azure_folder: str = typer.Option("azure_datasets", help="Output directory name/path for Azure CSVs"),
    container_name: Optional[str] = typer.Option(None, help="Azure Blob Storage Container Name"),
    sas_expiry_days: int = typer.Option(30, help="Days until SAS URL expires"),
):
    """Upload dataset images to Azure Blob Storage and generate an Azure CSV with SAS URLs.

    The output CSV is saved in `azure_datasets/` with the exact same filename as the original input CSV
    (or <dataset_name>.csv for HydraVision datasets).
    """
    from datetime import datetime, timedelta, timezone
    from dotenv import load_dotenv

    # Handle Typer OptionInfo when called directly as Python function
    if not isinstance(azure_folder, str):
        azure_folder = getattr(azure_folder, "default", "azure_datasets")
    if not isinstance(container_name, str):
        container_name = getattr(container_name, "default", None)
    if not isinstance(sas_expiry_days, int):
        try:
            sas_expiry_days = int(getattr(sas_expiry_days, "default", 30))
        except Exception:
            sas_expiry_days = 30

    # Load credentials from ~/.dltk.config or environment
    load_dotenv(Path.home() / ".dltk.config")

    try:
        from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
    except ImportError:
        logger.error("azure-storage-blob is not installed. Please run `uv pip install azure-storage-blob`.")
        sys.exit(1)

    cfg = OmegaConf.load(str(config_file))
    dataset_cfg = cfg.get("dataset", {})
    dsttype = dataset_cfg.get("type", "flat csv").lower()

    if dsttype == "hydravision":
        import data_factory.client.hydravision as hv
        dataset_name = dataset_cfg.get("dataset_name")
        if not dataset_name:
            raise ValueError("dataset_name is required for hydravision dataset type.")
        logger.info("Retrieving HydraVision dataset: {}", dataset_name)
        df = hv.HydraVisionGetDataset(dataset_name).read_dataframe()
        original_filename = f"{dataset_name}.csv"
    else:
        path = dataset_cfg.get("path")
        if not path:
            raise ValueError("path is required for flat csv dataset type.")
        logger.info("Reading flat CSV from {}", path)
        df = pd.read_csv(path)
        original_filename = Path(path).name

    # Target output directory & file in azure_datasets
    output_dir = REPO_ROOT / azure_folder if not Path(azure_folder).is_absolute() else Path(azure_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_csv_path = output_dir / original_filename

    image_col = None
    for col in ("im_url", "image_url", "url", "image", "original_url"):
        if col in df.columns:
            image_col = col
            break

    if not image_col:
        raise ValueError(f"Could not find image URL column in dataset. Available: {list(df.columns)}")

    # Azure credentials setup
    conn_str = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
    account_name = os.environ.get("AZURE_STORAGE_ACCOUNT_NAME")
    account_key = os.environ.get("AZURE_STORAGE_ACCOUNT_KEY")
    container = container_name or os.environ.get("AZURE_STORAGE_CONTAINER", "dataset-cache")

    if conn_str:
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        account_name = blob_service_client.account_name
        account_key = blob_service_client.credential.account_key
    elif account_name and account_key:
        blob_service_client = BlobServiceClient(
            account_url=f"https://{account_name}.blob.core.windows.net",
            credential=account_key
        )
    else:
        raise ValueError(
            "Azure credentials not found. Please set AZURE_STORAGE_CONNECTION_STRING or "
            "AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY in ~/.dltk.config or environment."
        )

    container_client = blob_service_client.get_container_client(container)
    if not container_client.exists():
        logger.info("Creating Azure Blob container: {}", container)
        container_client.create_container()

    urls = df[image_col].dropna().unique()
    logger.info("Preparing and uploading {} images to Azure Blob container '{}'...", len(urls), container)

    url_to_sas = {}
    success_count = 0

    for idx, url in enumerate(urls):
        url_str = str(url).strip()
        if not url_str:
            continue

        if "blob.core.windows.net" in url_str:
            url_to_sas[url_str] = url_str
            success_count += 1
            continue

        blob_name = f"images/{_stable_image_id(url_str)}.jpg"
        blob_client = container_client.get_blob_client(blob_name)

        try:
            if not blob_client.exists():
                image_bytes = get_image_content(url_str, timeout=30)
                if image_bytes is None:
                    logger.warning("Failed to fetch image bytes for {}. Keeping original URL.", url_str)
                    url_to_sas[url_str] = url_str
                    continue
                blob_client.upload_blob(image_bytes, overwrite=True)

            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=container,
                blob_name=blob_name,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(days=sas_expiry_days)
            )
            sas_url = f"https://{account_name}.blob.core.windows.net/{container}/{blob_name}?{sas_token}"
            url_to_sas[url_str] = sas_url
            success_count += 1
        except Exception as err:
            logger.error("Error processing/uploading image {}: {}", url_str, err)
            url_to_sas[url_str] = url_str

        if (idx + 1) % 50 == 0 or (idx + 1) == len(urls):
            logger.info("Processed [{}/{}] images for Azure", idx + 1, len(urls))

    df[image_col] = df[image_col].map(lambda u: url_to_sas.get(str(u).strip(), u))

    df.to_csv(target_csv_path, index=False)
    logger.info(
        "Successfully generated Azure CSV with {} SAS image URLs at: {}",
        success_count,
        target_csv_path
    )


if __name__ == "__main__":
    app()
