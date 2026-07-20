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
from dotenv import load_dotenv
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field

# Load environment credentials from ~/.dltk.config or .env
_dltk_config = Path.home() / ".dltk.config"
if _dltk_config.exists():
    load_dotenv(_dltk_config)
load_dotenv()

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from detection.geometry import compute_iou
from utils.vis_image import get_image_content
from llm.executor import LLMExecutor
from llm.utils import resolve_model_alias

app = typer.Typer(pretty_exceptions_show_locals=False)


class VerificationResult(BaseModel):
    is_valid: bool = Field(..., description="Whether the crop contains the specified class of accessory or item.")
    reason: str = Field(..., description="Short explanation of the validation result.")

    def __bool__(self) -> bool:
        return self.is_valid


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


class PredictionCache:
    """Persistent JSON cache for model predictions (detector outputs & LLM verification)."""

    def __init__(self, cache_file: Path, enabled: bool = True):
        self.cache_file = Path(cache_file)
        self.enabled = enabled
        self._data: dict[str, Any] = {}
        if self.enabled and self.cache_file.exists():
            try:
                self._data = json.loads(self.cache_file.read_text())
                logger.info("Loaded {} cached prediction entries from {}", len(self._data), self.cache_file)
            except Exception as e:
                logger.warning("Failed to load prediction cache from {}: {}", self.cache_file, e)

    @property
    def cache(self) -> dict[str, Any]:
        return self._data

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self._data[key] = value

    def save(self) -> None:
        if not self.enabled:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            self.cache_file.write_text(json.dumps(self._data, indent=2))
        except Exception as e:
            logger.warning("Failed to save prediction cache to {}: {}", self.cache_file, e)



def safe_format(template: str, **kwargs) -> str:
    for k, v in kwargs.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def normalize_execution_mode(mode: Any) -> str:
    if not isinstance(mode, str):
        mode = getattr(mode, "default", "full")
    normalized = str(mode or "").strip().lower()
    allowed = {"full", "detection_only"}
    if normalized not in allowed:
        raise ValueError(f"Invalid mode '{mode}'. Allowed modes: {sorted(allowed)}")
    return normalized


def parse_existing_boxes_and_concepts(row: pd.Series) -> List[Dict[str, Any]]:
    if "boxes" not in row.index or "concepts" not in row.index:
        logger.debug("Row is missing 'boxes' or 'concepts' column; skipping existing detection parsing.")
        return []

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


def _upscale_crop_if_too_small(
    crop: np.ndarray,
    min_short_side: int,
    min_long_side: int,
    interpolation: int,
) -> np.ndarray:
    """Upscale tiny crops so the LLM can inspect clearer details.

    This keeps aspect ratio and only upscales (never downsamples).
    """
    if crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    short_side = min(h, w)
    long_side = max(h, w)

    # No resize needed if the crop already satisfies the configured minima.
    if short_side >= min_short_side and long_side >= min_long_side:
        return crop

    # Ensure valid, non-zero targets.
    target_short = max(1, int(min_short_side))
    target_long = max(1, int(min_long_side))

    scale_short = target_short / short_side if short_side > 0 else 1.0
    scale_long = target_long / long_side if long_side > 0 else 1.0
    scale = max(1.0, scale_short, scale_long)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(crop, (new_w, new_h), interpolation=interpolation)


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

            # Map API label back to the original class name (which may have underscores).
            # class_to_prompt keys are space-separated; values are the raw underscore form.
            label_norm = label.replace(" ", "_").lower()
            mapped_label = class_to_prompt.get(label)  # exact match (space form)
            if mapped_label is None:
                # Fallback: iterate for case-insensitive / underscore-normalised match
                for pk, vk in class_to_prompt.items():
                    if pk.lower() == label or pk.replace(" ", "_").lower() == label_norm:
                        mapped_label = vk
                        break
            if mapped_label is None:
                mapped_label = label  # last resort: use as-is

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

    # class_aliases: maps a canonical output class to a list of query terms to send to the API.
    # e.g.  hair_accessories: [hijab, headband, headscarf, bandana, large hairband]
    # The API is queried with the alias terms; returned detections are remapped to the canonical name.
    class_aliases: dict = model_cfg.get("class_aliases", {})
    alias_to_canonical: dict = {}
    for canonical, aliases in class_aliases.items():
        for alias in (aliases or []):
            alias_norm = str(alias).strip() if alias is not None else ""
            if not alias_norm:
                continue  # skip blank entries from YAML e.g. "- "
            alias_to_canonical[alias_norm] = canonical
            alias_to_canonical[alias_norm.replace(" ", "_")] = canonical
    if class_aliases:
        logger.debug("class_aliases active: {}", class_aliases)

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

    # Expand canonical names that have aliases: replace the canonical name in each group
    # with the alias query terms so the API is prompted with the specific subcategories.
    if class_aliases:
        expanded_groups = []
        for group in groups:
            expanded_group = []
            for cls in group:
                if cls in class_aliases:
                    expanded_group.extend(class_aliases[cls])
                else:
                    expanded_group.append(cls)
            expanded_groups.append(expanded_group)
        groups = expanded_groups

    all_dets = []
    for group in groups:
        group_dets = _query_locate_anything_api(image_path, group, endpoint_url, decoding_mode)
        all_dets.extend(group_dets)

    # Remap alias detections back to their canonical class name
    if alias_to_canonical:
        for det in all_dets:
            name = det["name"]
            canonical = alias_to_canonical.get(name) or alias_to_canonical.get(name.replace("_", " "))
            if canonical:
                logger.debug("Remapping '{}' -> '{}'", name, canonical)
                det["name"] = canonical

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

    # Use an MD5-based cache filename so this is consistent with the blob names
    # written by `prep-azure` (which also uses _stable_image_id for blob paths).
    # This avoids the slug-based collision where two different URLs sharing the
    # same path stem would overwrite each other in cache.
    md5_cache_path = cache_dir / f"{_stable_image_id(im_url)}.jpg"
    if md5_cache_path.exists():
        try:
            image_bytes = md5_cache_path.read_bytes()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                return image, md5_cache_path
        except Exception:
            pass

    # Fetch image bytes and cache with the MD5-based filename
    image_bytes = get_image_content(im_url, timeout=20)
    if image_bytes is None:
        logger.warning("Could not download image from {}", im_url)
        return None, None
    try:
        md5_cache_path.write_bytes(image_bytes)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Could not decode image from {}", im_url)
            return None, None
        return image, md5_cache_path
    except Exception as exc:
        logger.warning("Could not process image from {}: {}", im_url, exc)
        return None, None


def validate_detection_with_llm(
    image: np.ndarray,
    det: dict,
    llm_cfg: dict,
    executor,
    system_prompt: str,
    user_prompt_tmpl: str,
    im_id: str = "",
    pred_cache: Optional[PredictionCache] = None,
) -> bool:
    class_name = det["name"]
    box = det.get("box", [0, 0, 0, 0])
    box_str = f"[{int(round(box[0]))},{int(round(box[1]))},{int(round(box[2]))},{int(round(box[3]))}]"

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
    min_crop_short_side = int(get_cfg_val("min_crop_short_side", 192))
    min_crop_long_side = int(get_cfg_val("min_crop_long_side", 256))
    jpeg_quality = int(get_cfg_val("jpeg_quality", 98))

    interpolation_name = str(get_cfg_val("upscale_interpolation", "cubic")).strip().lower()
    interpolation_map = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "area": cv2.INTER_AREA,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
    }
    interpolation = interpolation_map.get(interpolation_name, cv2.INTER_CUBIC)

    # Keep quality in OpenCV-supported bounds.
    jpeg_quality = max(1, min(100, jpeg_quality))

    # Crop the box
    crop = prepare_crop(image, det["box"], padding=padding, contrast=contrast, brightness=brightness)
    if crop.size == 0:
        logger.warning("Empty crop for detection of class {}, skipping LLM validation", class_name)
        return True

    crop = _upscale_crop_if_too_small(
        crop,
        min_short_side=min_crop_short_side,
        min_long_side=min_crop_long_side,
        interpolation=interpolation,
    )

    # Encode crop to JPEG bytes
    success, encoded_img = cv2.imencode(
        ".jpg",
        crop,
        [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
    )
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

    # Include effective prompt + preprocessing settings in cache identity.
    # This ensures cache invalidation when crop quality knobs are changed.
    cache_signature = {
        "class_name": class_name,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "padding": padding,
        "contrast": contrast,
        "brightness": brightness,
        "min_crop_short_side": min_crop_short_side,
        "min_crop_long_side": min_crop_long_side,
        "upscale_interpolation": interpolation_name,
        "jpeg_quality": jpeg_quality,
        "llm_model": str(llm_cfg.get("model_type", "")),
    }
    prompt_hash = hashlib.md5(json.dumps(cache_signature, sort_keys=True, default=str).encode()).hexdigest()[:12]
    cache_key = f"llm:{im_id}:{class_name}:{box_str}:{prompt_hash}"

    if pred_cache:
        cached_res = pred_cache.get(cache_key)
        if cached_res is not None:
            is_val = bool(cached_res.get("is_valid"))
            reason_str = str(cached_res.get("reason", "") or "")
            logger.info("LLM validation for {} (cached): is_valid={}, reason={}", class_name, is_val, reason_str)
            return VerificationResult(is_valid=is_val, reason=reason_str)

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
        is_valid = bool(parsed.is_valid)
        reason_str = str(getattr(parsed, "reason", "") or "")

        # If LLM rejects due to blurriness, accept detection and flag crop as blurry
        blurry_keywords = ["blurry", "blurred", "lacks clear detail", "blurry and lacks", "out of focus"]
        if not is_valid and any(kw in reason_str.lower() for kw in blurry_keywords):
            is_valid = True
            reason_str = f"[FLAGGED BLURRY] {reason_str}"
            logger.info("LLM crop for {} is blurry — accepting detection and flagging.", class_name)

        if pred_cache:
            pred_cache.set(cache_key, {"is_valid": is_valid, "reason": reason_str})
        logger.info("LLM validation for {}: is_valid={}, reason={}", class_name, is_valid, reason_str)
        return VerificationResult(is_valid=is_valid, reason=reason_str)
    except Exception as e:
        logger.error("LLM validation call failed for class {} — dropping detection to fail safe: {}", class_name, e)
        return VerificationResult(is_valid=False, reason=f"LLM call failed: {e}")


def resolve_dataset_csv(dataset_cfg: dict, prefer_azure: bool = True) -> tuple[pd.DataFrame, str]:
    """Resolve and read dataset DataFrame from HydraVision or flat CSV.

    Checks:
    1. azure_datasets/<original_filename>.csv (if prefer_azure=True and file exists)
    2. REPO_ROOT / path (if relative path exists relative to project root)
    3. Path(path) (if absolute or cwd-relative path exists)
    4. Fallback to azure_datasets/<original_filename>.csv if raw path does not exist
    """
    dsttype = dataset_cfg.get("type", "flat csv").lower()
    if dsttype == "hydravision":
        import data_factory.client.hydravision as hv
        dataset_name = dataset_cfg.get("dataset_name")
        if not dataset_name:
            raise ValueError("dataset_name is required for hydravision dataset type.")
        logger.info("Retrieving HydraVision dataset: {}", dataset_name)
        df = hv.HydraVisionGetDataset(dataset_name).read_dataframe()
        return df, f"{dataset_name}.csv"

    path = dataset_cfg.get("path")
    if not path:
        raise ValueError("path is required for flat csv dataset type.")

    original_filename = Path(path).name
    azure_folder = dataset_cfg.get("azure_folder", "azure_datasets")
    azure_csv = REPO_ROOT / azure_folder / original_filename
    repo_path = REPO_ROOT / path
    direct_path = Path(path)

    if prefer_azure and azure_csv.exists():
        logger.info("Azure dataset found at {}. Using SAS URLs instead of original path {}.", azure_csv, path)
        target = azure_csv
    elif repo_path.exists():
        logger.info("Reading flat CSV from {}", repo_path)
        target = repo_path
    elif direct_path.exists():
        logger.info("Reading flat CSV from {}", direct_path)
        target = direct_path
    elif azure_csv.exists():
        logger.info("Original path {} not found, but Azure dataset found at {}. Using Azure dataset.", path, azure_csv)
        target = azure_csv
    else:
        raise FileNotFoundError(
            f"Dataset CSV not found for path '{path}'. "
            f"Checked relative to repo root ({repo_path}), direct path ({direct_path}), "
            f"and Azure dataset fallback ({azure_csv})."
        )

    df = pd.read_csv(target)
    return df, original_filename


@app.command()
def main(
    config_file: str = typer.Argument(..., help="Path to auto-annotate config YAML"),
    mode: str = typer.Option(
        "full",
        "--mode",
        help="Execution mode: 'full' runs detection+LLM, 'detection_only' runs detection+dedup only.",
    ),
):
    """Run the auto-annotation curation pipeline."""
    cfg = OmegaConf.load(str(config_file))
    execution_mode = normalize_execution_mode(mode)
    logger.info("Execution mode: {}", execution_mode)

    dataset_cfg = cfg.get("dataset", {})
    dsttype = dataset_cfg.get("type", "flat csv").lower()

    # 1. Retrieve dataset
    df, original_filename = resolve_dataset_csv(dataset_cfg, prefer_azure=True)

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

    pred_cache_file = cache_dir / "predictions_cache.json"
    use_pred_cache = bool(dataset_cfg.get("use_prediction_cache", True))
    pred_cache = PredictionCache(pred_cache_file, enabled=use_pred_cache)

    # 2. Setup LLM if validation is enabled for this execution mode
    llm_cfg = cfg.get("llm_validation", cfg.get("llm validation"))
    llm_executor = None
    system_prompt = ""
    user_prompt_tmpl = ""
    llm_classes = []

    if execution_mode == "full" and llm_cfg:
        # Convert from OmegaConf DictConfig to a plain dict so that .get() calls
        # with mutable defaults (e.g. {}) are safe throughout the rest of the pipeline.
        llm_cfg = OmegaConf.to_container(llm_cfg, resolve=True)
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
    elif execution_mode == "detection_only":
        logger.info("Detection-only mode active. Skipping LLM validation stage.")

    # 3. Process each row
    detection_models = cfg.get("detection_models", cfg.get("detection models", []))
    if isinstance(detection_models, dict):
        # Support dict format as single model or dict of models
        if "model_type" in detection_models:
            detection_models = [detection_models]
        else:
            detection_models = list(detection_models.values())
    # Convert each model config from OmegaConf to plain dict
    detection_models = [
        OmegaConf.to_container(m, resolve=True) if hasattr(m, "_metadata") else dict(m)
        for m in detection_models
    ]

    # Validate class_groups vs classes consistency
    for m in detection_models:
        class_groups = m.get("class_groups")
        classes = m.get("classes", [])
        if class_groups:
            grouped = [c for g in class_groups for c in g]
            extra_in_groups = set(grouped) - set(classes)
            extra_in_classes = set(classes) - set(grouped)
            if extra_in_groups or extra_in_classes:
                logger.warning(
                    "class_groups and classes are out of sync for model '{}'. "
                    "In groups but not classes: {}. In classes but not groups: {}.",
                    m.get("model_type", "?"), sorted(extra_in_groups), sorted(extra_in_classes)
                )

    keep_bbs = bool(dataset_cfg.get("keep_bounding_boxes", False))
    dedup_cfg = cfg.get("dedup_policy", cfg.get("dedup policy", {}))
    if dedup_cfg:
        # Convert from OmegaConf to plain dict so .get(cls_name) is safe for all class names
        dedup_cfg = OmegaConf.to_container(dedup_cfg, resolve=True)

    annotated_rows = []
    viz_items = []
    logger.info("Processing {} images...", len(df))

    for i, (idx, row) in enumerate(df.iterrows(), 1):
        im_url = row["im_url"]
        im_id = row.get("im_id", _stable_image_id(im_url))
        logger.info("Processing image [{}/{}]: {}", i, len(df), im_url)

        # Download/read image
        image, local_path = load_image_and_path(im_url, cache_dir)
        if image is None:
            # Preserve the row in the output (with NaN annotations) so the output
            # CSV stays aligned with the input and downstream positional joins don't break.
            logger.warning("Failed to load image from {}. Row kept with empty annotations.", im_url)
            row_updated = dict(row)
            row_updated["concepts"] = None
            row_updated["boxes"] = None
            annotated_rows.append(row_updated)
            viz_items.append({
                "im_id": im_id,
                "im_url": im_url,
                "detections": [],
            })
            continue

        # Start with existing boxes if configured
        row_dets = []
        if keep_bbs:
            row_dets.extend(parse_existing_boxes_and_concepts(row))

        # Run configured detectors
        for model_cfg in detection_models:
            model_type = model_cfg.get("model_type")
            cfg_hash = hashlib.md5(json.dumps(model_cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]
            det_cache_key = f"det:{im_id}:{model_type}:{cfg_hash}"

            cached_dets = pred_cache.get(det_cache_key)
            if cached_dets is not None:
                logger.debug("Reusing cached {} detections for image [{}]", model_type, im_url)
                row_dets.extend(cached_dets)
            else:
                m_dets = []
                if model_type == "fashion_model":
                    try:
                        m_dets = run_fashion_model(image, model_cfg)
                    except Exception as e:
                        logger.error("Fashion model execution failed: {}", e)
                elif model_type == "locate_anything":
                    if local_path is None:
                        logger.warning("local_path is None for {}; skipping locate_anything.", im_url)
                    else:
                        try:
                            m_dets = run_locate_anything(str(local_path), model_cfg)
                        except Exception as e:
                            logger.error("Locate anything execution failed: {}", e)
                pred_cache.set(det_cache_key, m_dets)
                row_dets.extend(m_dets)

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

        # 5. Apply LLM validation (with candidate box count safety caps)
        viz_detections = []
        if llm_executor and row_dets:
            max_per_img = int(llm_cfg.get("max_boxes_per_image", 20))
            max_per_cls = int(llm_cfg.get("max_boxes_per_class", 6))

            # Cap candidate boxes per class per image first
            capped_dets = []
            dets_by_cls = defaultdict(list)
            for d in row_dets:
                dets_by_cls[d["name"]].append(d)

            for cls_name, cls_dets in dets_by_cls.items():
                if len(cls_dets) > max_per_cls:
                    logger.warning(
                        "Image [{}] has {} candidate boxes for class '{}', exceeding max_boxes_per_class ({}). Truncating to top {}.",
                        im_url, len(cls_dets), cls_name, max_per_cls, max_per_cls
                    )
                    cls_dets = sorted(cls_dets, key=lambda x: float(x.get("score", 1.0)), reverse=True)[:max_per_cls]
                capped_dets.extend(cls_dets)

            # Cap total candidate boxes per image
            if len(capped_dets) > max_per_img:
                logger.warning(
                    "Image [{}] has {} total candidate boxes, exceeding max_boxes_per_image ({}). Truncating to top {}.",
                    im_url, len(capped_dets), max_per_img, max_per_img
                )
                capped_dets = sorted(capped_dets, key=lambda x: float(x.get("score", 1.0)), reverse=True)[:max_per_img]

            # Check if safety cap is exceeded
            cap_exceeded = (len(row_dets) > max_per_img) or any(len(c_dets) > max_per_cls for c_dets in dets_by_cls.values())
            if cap_exceeded:
                logger.warning(
                    "Image [{}] has {} candidate boxes exceeding safety cap (max_per_img={}, max_per_cls={}). Skipping LLM validation and flagging image.",
                    im_url, len(row_dets), max_per_img, max_per_cls
                )
                capped_dets = sorted(row_dets, key=lambda x: float(x.get("score", 1.0)), reverse=True)[:max_per_img]
                for d in capped_dets:
                    det_viz = dict(d)
                    det_viz["llm_validated"] = False
                    det_viz["is_valid"] = True
                    det_viz["reason"] = "[FLAGGED CAP EXCEEDED] Candidate box safety cap exceeded; LLM validation bypassed."
                    viz_detections.append(det_viz)
                row_dets = capped_dets
            else:
                row_dets = capped_dets
                validated_dets = []
                for d in row_dets:
                    # Check if we should validate this class
                    should_validate = (not llm_classes) or (d["name"].lower() in llm_classes)
                    det_viz = dict(d)
                    if should_validate:
                        res = validate_detection_with_llm(
                            image, d, llm_cfg, llm_executor, system_prompt, user_prompt_tmpl,
                            im_id=im_id, pred_cache=pred_cache
                        )
                        is_val = bool(getattr(res, "is_valid", res))
                        reason_str = str(getattr(res, "reason", "") or "")
                        det_viz["llm_validated"] = True
                        det_viz["is_valid"] = is_val
                        det_viz["reason"] = reason_str
                        if is_val:
                            validated_dets.append(d)
                    else:
                        det_viz["llm_validated"] = False
                        det_viz["is_valid"] = True
                        det_viz["reason"] = ""
                        validated_dets.append(d)
                    viz_detections.append(det_viz)
                row_dets = validated_dets
        else:
            for d in row_dets:
                det_viz = dict(d)
                det_viz["llm_validated"] = False
                det_viz["is_valid"] = True
                det_viz["reason"] = ""
                viz_detections.append(det_viz)

        viz_items.append({
            "im_id": im_id,
            "im_url": im_url,
            "detections": viz_detections,
        })

        # Save prediction cache after processing image
        pred_cache.save()

        # Form final row values
        # Use None (→ NaN in the CSV) rather than an empty string for missing
        # detections so downstream column parsers can distinguish "no data" from
        # "zero detections explicitly annotated".
        concepts_str = ",".join([d["name"] for d in row_dets]) if row_dets else None
        boxes_str = (
            ",".join(
                [f"[{int(round(b[0]))},{int(round(b[1]))},{int(round(b[2]))},{int(round(b[3]))}]"
                 for b in [d["box"] for d in row_dets]]
            ) if row_dets else None
        )

        row_updated = dict(row)
        row_updated["concepts"] = concepts_str
        row_updated["boxes"] = boxes_str
        annotated_rows.append(row_updated)

    # 6. Save results — derive output filename from the input source so repeated
    # runs don't silently overwrite each other.
    if dsttype == "hydravision":
        output_stem = dataset_cfg.get("dataset_name", "annotated_dataset")
    else:
        input_path = dataset_cfg.get("path", "annotated_dataset")
        output_stem = Path(input_path).stem
    output_suffix = "_annotated.csv" if execution_mode == "full" else "_detections_only.csv"
    output_path = output_dir / f"{output_stem}{output_suffix}"
    out_df = pd.DataFrame(annotated_rows)
    out_df.to_csv(output_path, index=False)
    logger.info(
        "Auto-annotation pipeline finished in mode '{}'. Saved {} rows to {}",
        execution_mode,
        len(out_df),
        output_path,
    )

    # 7. Generate HTML visualization
    try:
        from utils.html_visualization import generate_html_visualization
        viz_filename = dataset_cfg.get("visualization_filename", "visualization.html")
        html_out = output_dir / viz_filename
        generate_html_visualization(viz_items, html_out, title=f"Auto-Annotate Visualization — {output_stem}")
        logger.info("Generated HTML visualization gallery: {}", html_out)
    except Exception as e:
        logger.warning("Failed to generate HTML visualization: {}", e)


@app.command(name="visualize")
def generate_visualization_cmd(
    config_file: str = typer.Argument(..., help="Path to auto-annotate config YAML"),
):
    """Generate or refresh HTML visualization gallery from predictions cache & dataset."""
    from utils.html_visualization import generate_html_visualization

    cfg = OmegaConf.load(str(config_file))
    dataset_cfg = cfg.get("dataset", {})
    df, original_filename = resolve_dataset_csv(dataset_cfg, prefer_azure=True)

    output_dir = Path(dataset_cfg.get("output_dir", REPO_ROOT / "curated_datasets/curation"))
    custom_cache = dataset_cfg.get("cache_dir")
    cache_dir = Path(custom_cache) if custom_cache else output_dir / "cache"
    pred_cache_file = cache_dir / "predictions_cache.json"
    pred_cache = PredictionCache(pred_cache_file)

    image_col = None
    for col in ("im_url", "image_url", "url", "image", "original_url"):
        if col in df.columns:
            image_col = col
            break

    if not image_col:
        raise ValueError(f"Could not find image URL column in dataset.")

    keep_bbs = bool(dataset_cfg.get("keep_bounding_boxes", False))
    viz_items = []
    for _, row in df.iterrows():
        im_url = str(row[image_col]).strip()
        if not im_url:
            continue
        im_id = row.get("im_id", _stable_image_id(im_url))

        dets = []
        if keep_bbs:
            dets.extend(parse_existing_boxes_and_concepts(row))

        # Check predictions cache for detector keys
        for key, val in pred_cache.cache.items():
            if key.startswith(f"det:{im_id}:") and isinstance(val, list):
                dets.extend(val)

        # Apply deduplication policy across all detection sources (GT + detectors)
        dedup_cfg = cfg.get("dedup_policy", {})
        if dedup_cfg and dets:
            by_class = defaultdict(list)
            for d in dets:
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
                    deduped_cls_list = suppress_duplicates(cls_list, iou_thresh, io_min_thresh, keep_which)
                    deduped_dets.extend(deduped_cls_list)
                else:
                    deduped_dets.extend(cls_list)
            dets = deduped_dets

        # Check predictions cache for LLM validation entries
        viz_dets = []
        for d in dets:
            d_entry = dict(d)
            box = d.get("box", [0, 0, 0, 0])
            box_str = f"[{int(round(box[0]))},{int(round(box[1]))},{int(round(box[2]))},{int(round(box[3]))}]"
            cls_name = d.get("name", "")

            # Look up any LLM cache key for this box
            matched_llm = None
            for k, v in pred_cache.cache.items():
                if k.startswith(f"llm:{im_id}:{cls_name}:{box_str}:") and isinstance(v, dict):
                    matched_llm = v
                    break

            if matched_llm is not None:
                d_entry["llm_validated"] = True
                d_entry["is_valid"] = bool(matched_llm.get("is_valid"))
                d_entry["reason"] = str(matched_llm.get("reason", "") or "")
            else:
                d_entry["llm_validated"] = False
                d_entry["is_valid"] = True
                d_entry["reason"] = ""
            viz_dets.append(d_entry)

        viz_items.append({
            "im_id": im_id,
            "im_url": im_url,
            "detections": viz_dets,
        })

    output_stem = Path(original_filename).stem
    viz_filename = dataset_cfg.get("visualization_filename", "visualization.html")
    html_out = output_dir / viz_filename
    generate_html_visualization(viz_items, html_out, title=f"Auto-Annotate Visualization — {output_stem}")
    logger.info("Saved visualization HTML gallery ({} images) to {}", len(viz_items), html_out)


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
    df, original_filename = resolve_dataset_csv(dataset_cfg, prefer_azure=True)

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
    df, original_filename = resolve_dataset_csv(dataset_cfg, prefer_azure=False)

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
        if not account_key:
            raise ValueError(
                "Could not extract account_key from the connection string. "
                "SAS token generation requires a key-based connection string "
                "(not a SAS-based or Managed Identity connection string)."
            )
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
            if "?" not in url_str:
                logger.warning(
                    "URL {} looks like a private Azure Blob URL without a SAS token. "
                    "It will be written as-is and may be inaccessible downstream.",
                    url_str,
                )
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
