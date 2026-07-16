"""
RTMDetDetect: local Python implementation of the virecognition RTMDet detection pipeline.
"""

import json
import math
import ctypes
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

import cv2
import numpy as np
import onnxruntime as ort
import yaml


def _can_use_cuda_provider() -> bool:
    """Return True only when CUDA EP is available and key CUDA libs can be loaded."""
    try:
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
    except Exception:
        return False

    import sys
    lib_dirs = []
    for p in sys.path:
        if not p:
            continue
        nvidia_root = Path(p) / "nvidia"
        if nvidia_root.exists() and nvidia_root.is_dir():
            for lib_dir in nvidia_root.glob("*/lib"):
                if lib_dir.is_dir():
                    lib_dirs.append(lib_dir)

    for _ in range(3):
        for d in lib_dirs:
            for f in d.glob("lib*.so*"):
                if f.is_file() and not f.is_symlink():
                    try:
                        ctypes.CDLL(str(f), mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass

    for lib_name in ("libcublasLt.so.12", "libcudnn.so.9"):
        loaded = False
        try:
            ctypes.CDLL(lib_name, mode=ctypes.RTLD_GLOBAL)
            loaded = True
        except OSError:
            for d in lib_dirs:
                full_path = d / lib_name
                if full_path.exists():
                    try:
                        ctypes.CDLL(str(full_path), mode=ctypes.RTLD_GLOBAL)
                        loaded = True
                        break
                    except OSError:
                        pass
        if not loaded:
            return False
    return True


def _select_onnx_providers(
    providers: Optional[List[str]] = None,
) -> List[str]:
    """Choose stable providers and avoid noisy CUDA load errors when deps are missing."""
    if providers is not None:
        return providers

    if _can_use_cuda_provider():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    warnings.warn(
        "CUDAExecutionProvider is unavailable or CUDA runtime libraries are missing; "
        "falling back to CPUExecutionProvider.",
        RuntimeWarning,
    )
    return ["CPUExecutionProvider"]


# ---------------------------------------------------------------------------
# Step 1: image.color
# ---------------------------------------------------------------------------

def _image_color(image: np.ndarray, max_size: int = 1024) -> Tuple[np.ndarray, float]:
    """
    Replicates ImageColor::RunOnce.
    """
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        gray = image if image.ndim == 2 else image[:, :, 0]
        image = np.stack([gray, gray, gray], axis=2)

    h, w = image.shape[:2]
    scale = min(max_size / max(h, w), 1.0)
    if scale < 1.0:
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0

    return image.astype(np.uint8), scale


# ---------------------------------------------------------------------------
# Step 2: roi.detect_product — RTMDet
# ---------------------------------------------------------------------------

class _ModelConfig:
    """Parsed from model.cfg (YAML) and threshold.txt."""
    def __init__(self, model_dir: Path, threshold_file: str = "threshold.txt"):
        with open(model_dir / "model.cfg") as f:
            cfg = yaml.safe_load(f)
        self.mean = np.array(cfg["mean"], dtype=np.float32)          # (3,)
        self.std = np.array(cfg["std"], dtype=np.float32)            # (3,)
        self.img_scale: int = cfg.get("img_scale", 640)
        self.side_divisor: int = cfg.get("side_divisor", 32)
        self.max_per_im: int = cfg.get("max_per_im", 100)
        self.score_thr: float = cfg.get("score_thr", 0.05)
        self.num_classes: int = cfg.get("num_classes", 1)

        with open(model_dir / "model_info.json") as f:
            info = json.load(f)
        self.input_blob: str = info.get("input_blob", "data")

        # threshold.txt: class,threshold per line (ordered)
        self.thresh_classes: List[str] = []
        self.thresh_values: List[float] = []
        thresh_path = model_dir / threshold_file
        if thresh_path.exists():
            for line in thresh_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                name, val = line.split(",", 1)
                self.thresh_classes.append(name.strip())
                self.thresh_values.append(float(val.strip()))
        else:
            classes_path = model_dir / "classes.txt"
            if classes_path.exists():
                for line in classes_path.read_text().splitlines():
                    name = line.strip()
                    if not name:
                        continue
                    self.thresh_classes.append(name)
                    self.thresh_values.append(self.score_thr)


def _preprocess_centernet(
    image_rgb: np.ndarray,
    img_scale: int,
    side_divisor: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    RTMDet uses same preprocessing steps as CenterNet: resize to img_scale and normalize.
    """
    h, w = image_rgb.shape[:2]
    im_scale = (img_scale / h, img_scale / w)

    resized = cv2.resize(image_rgb, (img_scale, img_scale), interpolation=cv2.INTER_LINEAR)

    img_f = resized.astype(np.float32) / 255.0
    img_f = (img_f - mean) / std

    img_chw = img_f.transpose(2, 0, 1)

    _, rh, rw = img_chw.shape
    pad_h = math.ceil(rh / side_divisor) * side_divisor
    pad_w = math.ceil(rw / side_divisor) * side_divisor
    blob = np.zeros((1, 3, pad_h, pad_w), dtype=np.float32)
    blob[0, :, :rh, :rw] = img_chw

    return blob, im_scale


def _decode_rtmdet(
    cls_scores: List[np.ndarray],
    bbox_preds: List[np.ndarray],
    strides: List[int],
    orig_h: int,
    orig_w: int,
    im_scale: Tuple[float, float],
    score_thr: float,
    max_per_im: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Decodes multi-level RTMDet outputs into box coordinates and scores.
    """
    all_scores = []
    all_cls_idxs = []
    all_dets = []

    for cls_score, bbox_pred, stride in zip(cls_scores, bbox_preds, strides):
        _, num_classes, h_i, w_i = cls_score.shape

        # 1. Sigmoid class scores
        cls_score_flat = cls_score[0].reshape(num_classes, -1).T
        scores = 1.0 / (1.0 + np.exp(-np.clip(cls_score_flat, -100, 100)))

        # 2. Reshape bbox predictions: distance to boundary (l, t, r, b)
        bbox_pred_flat = bbox_pred[0].reshape(4, -1).T

        # 3. Generate grid centers (priors)
        px = (np.arange(w_i) + 0.5) * stride
        py = (np.arange(h_i) + 0.5) * stride
        shift_xx, shift_yy = np.meshgrid(px, py)
        grid_x = shift_xx.ravel()
        grid_y = shift_yy.ravel()

        # 4. Filter by score threshold
        max_scores = scores.max(axis=1)
        max_cls_idxs = scores.argmax(axis=1)

        keep_mask = max_scores >= score_thr
        if not np.any(keep_mask):
            continue

        scores_k = max_scores[keep_mask]
        cls_idxs_k = max_cls_idxs[keep_mask]
        bbox_pred_k = bbox_pred_flat[keep_mask]
        px_k = grid_x[keep_mask]
        py_k = grid_y[keep_mask]

        # Decode boxes: x1 = px - l, y1 = py - t, x2 = px + r, y2 = py + b
        x1 = px_k - bbox_pred_k[:, 0]
        y1 = py_k - bbox_pred_k[:, 1]
        x2 = px_k + bbox_pred_k[:, 2]
        y2 = py_k + bbox_pred_k[:, 3]

        # Map back to original image space
        x1 = np.clip(x1 / im_scale[1], 0, orig_w - 1)
        y1 = np.clip(y1 / im_scale[0], 0, orig_h - 1)
        x2 = np.clip(x2 / im_scale[1], 0, orig_w - 1)
        y2 = np.clip(y2 / im_scale[0], 0, orig_h - 1)

        dets_k = np.stack([x1, y1, x2, y2], axis=1)

        all_scores.append(scores_k)
        all_cls_idxs.append(cls_idxs_k)
        all_dets.append(dets_k)

    if len(all_scores) == 0:
        return np.array([]), np.array([]), np.zeros((0, 4))

    all_scores = np.concatenate(all_scores)
    all_cls_idxs = np.concatenate(all_cls_idxs)
    all_dets = np.concatenate(all_dets, axis=0)

    # Sort by score descending and keep top max_per_im
    order = np.argsort(-all_scores)
    order = order[:max_per_im]

    return all_scores[order], all_cls_idxs[order], all_dets[order]


def _nms(dets: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
    """Standard IoU-based greedy NMS. Returns kept indices."""
    if len(dets) == 0:
        return []
    x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1 + 1) * np.maximum(0, iy2 - iy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thresh]
    return keep


# ---------------------------------------------------------------------------
# Objectness scoring (SortFashionObjects / EvalFashionObjectObjectness)
# ---------------------------------------------------------------------------

def _score_size(image_rgb: np.ndarray, rois: List[Dict]) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    image_area = float(h * w)
    out = np.empty(len(rois), dtype=np.float32)
    for i, roi in enumerate(rois):
        x1, y1, x2, y2 = roi["box"]
        min_dim = min(x2 - x1 + 1, y2 - y1 + 1)
        out[i] = min_dim * min_dim / image_area
    return out


def _score_centerness(
    image_rgb: np.ndarray,
    rois: List[Dict],
    centerness_sigma: Dict[str, float],
) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    L = float(min(h, w))
    mu_y = (h + 1) / 2.0 / L
    mu_x = (w + 1) / 2.0 / L

    out = np.empty(len(rois), dtype=np.float32)
    for i, roi in enumerate(rois):
        x1, y1, x2, y2 = roi["box"]
        cx = (x1 + x2) / 2.0 / L
        cy = (y1 + y2) / 2.0 / L
        cat = roi["name"].split("-")[0]
        sigma = centerness_sigma.get(cat, 1.0 / 6.0)
        out[i] = math.exp(-0.5 * ((cy - mu_y) ** 2 + (cx - mu_x) ** 2) / (sigma ** 2))
    return out


def _score_cornerness(image_rgb: np.ndarray, rois: List[Dict]) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    scale = min(320.0 / max(h, w), 1.0)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    thresh = 5

    out = np.empty(len(rois), dtype=np.float32)
    for i, roi in enumerate(rois):
        x1, y1, x2, y2 = roi["box"]
        rx1 = x1 * scale
        ry1 = y1 * scale
        rx2 = x2 * scale
        ry2 = y2 * scale
        near_top = ry1 <= thresh
        near_left = rx1 <= thresh
        near_bottom = ry2 > new_h - thresh
        near_right = rx2 > new_w - thresh
        corner_count = (
            (near_top and near_left)
            + (near_bottom and near_right)
            + (near_top and near_right)
            + (near_bottom and near_left)
        )
        out[i] = 0.8 ** corner_count
    return out


def _sort_fashion_objects(
    image_rgb: np.ndarray,
    rois: List[Dict],
    centerness_sigma: Dict[str, float],
    objectness_prior: Dict[str, float],
) -> List[Dict]:
    if not rois:
        return rois

    det_scores = np.array([r["score"] for r in rois], dtype=np.float32)

    score_size = _score_size(image_rgb, rois)
    score_center = _score_centerness(image_rgb, rois, centerness_sigma)
    score_corner = _score_cornerness(image_rgb, rois)

    prior = np.array(
        [objectness_prior.get(r["name"].split("-")[0], 1.0) for r in rois],
        dtype=np.float32,
    )

    objectness = np.sqrt(
        score_size ** 0.25 * score_center ** 3.0 * score_corner ** 0.25
    ) * prior

    max_obj = objectness.max()
    if max_obj > 0:
        objectness /= max_obj

    new_scores = det_scores * objectness

    order = np.argsort(-new_scores)
    result = []
    for idx in order:
        r = dict(rois[idx])
        r["score"] = float(new_scores[idx])
        result.append(r)
    return result


# ---------------------------------------------------------------------------
# box_concept / category filtering
# ---------------------------------------------------------------------------

_CATEGORY_ALL = "all"
_CATEGORY_OTHER = "other"
_CATEGORY_WHOLEIMAGE = "wholeimage"

_CATEGORY_EXPANSIONS: Dict[str, List[str]] = {
    "bottom":      ["ethnic_wear", "skirt"],
    "dress":       ["ethnic_wear", "outerwear", "skirt", "top"],
    "ethnic_wear": ["dress", "skirt", "top"],
    "outerwear":   ["dress", "top"],
    "skirt":       ["bottom", "dress", "ethnic_wear"],
    "top":         ["dress", "ethnic_wear", "outerwear"],
}


def _validate_object_type(box_concept: str, thresh_classes: List[str]) -> str:
    if not box_concept:
        return _CATEGORY_OTHER
    parts = [t.strip() for t in box_concept.split(";") if t.strip()]
    valid: List[str] = []
    for t in parts:
        if t in (_CATEGORY_ALL, _CATEGORY_WHOLEIMAGE):
            return t
        if t in thresh_classes or t == _CATEGORY_OTHER:
            valid.append(t)
    return ";".join(valid) if valid else _CATEGORY_OTHER


def _pick_category_indices(object_type: str, thresh_classes: List[str]) -> Optional[set]:
    if not object_type or object_type in (_CATEGORY_ALL, _CATEGORY_OTHER, _CATEGORY_WHOLEIMAGE):
        return None

    cat_idx_map = {name: i for i, name in enumerate(thresh_classes)}
    parts = [t.strip() for t in object_type.split(";") if t.strip()]
    auto_expand = (len(parts) == 1)

    indices: set = set()
    for t in parts:
        if t in cat_idx_map:
            indices.add(cat_idx_map[t])
            if auto_expand and t in _CATEGORY_EXPANSIONS:
                for expanded in _CATEGORY_EXPANSIONS[t]:
                    if expanded in cat_idx_map:
                        indices.add(cat_idx_map[expanded])
    return indices if indices else None


def _parse_output_type(object_type: str) -> str:
    if not object_type:
        return ""
    output_type = ""
    if "/" in object_type:
        head, output_type = object_type.split("/", 1)
        parts = [t.strip() for t in head.split(";") if t.strip()]
    else:
        parts = [t.strip() for t in object_type.split(";") if t.strip()]
    if not output_type and len(parts) == 1:
        output_type = parts[0]
    return output_type


# ---------------------------------------------------------------------------
# Fallback detection
# ---------------------------------------------------------------------------

def _get_fashion_object_simple(image_rgb: np.ndarray) -> List[Dict[str, Any]]:
    h, w = image_rgb.shape[:2]
    scale = min(300.0 / max(h, w), 1.0)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    small = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    mask = cv2.Canny(gray, 60, 150)
    pts = cv2.findNonZero(mask)
    if pts is not None:
        x_mn = int(pts[:, 0, 0].min() / scale)
        y_mn = int(pts[:, 0, 1].min() / scale)
        x_mx = int(pts[:, 0, 0].max() / scale)
        y_mx = int(pts[:, 0, 1].max() / scale)
        if (x_mx - x_mn) >= 20 and (y_mx - y_mn) >= 20:
            return [{"name": _CATEGORY_OTHER, "score": 100.0, "box": [x_mn, y_mn, x_mx, y_mx]}]
    return [{"name": _CATEGORY_OTHER, "score": 100.0, "box": [0, 0, w - 1, h - 1]}]


def _score_color_contrast_eb(
    image_rgb: np.ndarray,
    rois: List[Dict],
) -> np.ndarray:
    if not rois:
        return np.array([], dtype=np.float32)
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w = image_lab.shape[:2]
    theta = 100.0
    scores = np.empty(len(rois), dtype=np.float32)
    for i, roi in enumerate(rois):
        x1 = int(np.clip(roi["box"][0], 0, w - 1))
        y1 = int(np.clip(roi["box"][1], 0, h - 1))
        x2 = int(np.clip(roi["box"][2], 0, w - 1))
        y2 = int(np.clip(roi["box"][3], 0, h - 1))
        if x2 <= x1 or y2 <= y1:
            scores[i] = 0.1
            continue
        interior_mean = image_lab[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0)
        band = max(1, int(min(x2 - x1, y2 - y1) * 0.1))
        border = np.concatenate([
            image_lab[max(0, y1 - band):y1, x1:x2].reshape(-1, 3),
            image_lab[y2:min(h, y2 + band), x1:x2].reshape(-1, 3),
            image_lab[y1:y2, max(0, x1 - band):x1].reshape(-1, 3),
            image_lab[y1:y2, x2:min(w, x2 + band)].reshape(-1, 3),
        ], axis=0)
        border_mean = border.mean(axis=0) if len(border) > 0 else interior_mean
        scores[i] = float(np.clip(np.linalg.norm(interior_mean - border_mean) / theta, 0.1, 1.0))
    return scores


def _detect_edgebox(
    image_rgb: np.ndarray,
    edgebox_model_path: Optional[str] = None,
    area_low: float = 0.05,
    area_high: float = 0.95,
    max_boxes: int = 200,
    edgebox_always_return: bool = False,
) -> List[Dict[str, Any]]:
    orig_h, orig_w = image_rgb.shape[:2]
    scale = min(320.0 / max(orig_h, orig_w), 1.0)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))

    if new_h < 20 or new_w < 20:
        return [{"name": _CATEGORY_OTHER, "score": 1.0, "box": [0, 0, orig_w - 1, orig_h - 1]}]

    small = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    edges: Optional[np.ndarray] = None
    orimap: Optional[np.ndarray] = None
    if edgebox_model_path and Path(edgebox_model_path).exists():
        try:
            edge_det = cv2.ximgproc.createStructuredEdgeDetection(edgebox_model_path)
            small_f = small.astype(np.float32) / 255.0
            edges = edge_det.detectEdges(small_f)
            orimap = edge_det.computeOrientation(edges)
            edges = edge_det.edgesNms(edges, orimap)
        except Exception:
            edges = None
    if edges is None:
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 60, 150).astype(np.float32) / 255.0
        gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        orimap = np.arctan2(gy, gx).astype(np.float32)

    try:
        eb = cv2.ximgproc.createEdgeBoxes(maxBoxes=max_boxes)
        raw = eb.getBoundingBoxes(edges, orimap)
        if raw is None or (isinstance(raw, tuple) and raw[0] is None):
            boxes_xywh, eb_scores = None, None
        elif isinstance(raw, tuple) and len(raw) == 2:
            boxes_xywh, eb_scores = raw
        else:
            boxes_xywh, eb_scores = raw, None
    except Exception:
        boxes_xywh, eb_scores = None, None

    if boxes_xywh is None or len(boxes_xywh) == 0:
        if edgebox_always_return:
            return [{"name": _CATEGORY_OTHER, "score": 1.0, "box": [0, 0, orig_w - 1, orig_h - 1]}]
        return []

    image_area = float(new_h * new_w)
    valid_mask = []
    for x, y, bw, bh in boxes_xywh:
        ratio = (bw * bh) / image_area
        valid_mask.append(area_low < ratio < area_high)
    valid_mask = np.array(valid_mask, dtype=bool)
    boxes_xywh = boxes_xywh[valid_mask]
    if eb_scores is not None:
        eb_scores = eb_scores.ravel()[valid_mask]

    if len(boxes_xywh) == 0:
        if edgebox_always_return:
            return [{"name": _CATEGORY_OTHER, "score": 1.0, "box": [0, 0, orig_w - 1, orig_h - 1]}]
        return []

    if eb_scores is not None and eb_scores.max() > 0:
        eb_scores = eb_scores.astype(np.float32) / eb_scores.max()
    else:
        eb_scores = np.ones(len(boxes_xywh), dtype=np.float32)

    rois_orig: List[Dict] = []
    for j, (x, y, bw, bh) in enumerate(boxes_xywh):
        rois_orig.append({
            "name": _CATEGORY_OTHER,
            "score": float(eb_scores[j]),
            "box": [int(x / scale), int(y / scale),
                    int((x + bw) / scale), int((y + bh) / scale)],
        })

    color_scores = _score_color_contrast_eb(image_rgb, rois_orig)
    final_scores = eb_scores * (color_scores ** 0.25)
    for j in range(len(rois_orig)):
        rois_orig[j]["score"] = float(final_scores[j])

    rois_orig.sort(key=lambda r: -r["score"])
    return rois_orig


def _roi_detect_product(
    image_rgb: np.ndarray,
    sess: ort.InferenceSession,
    cfg: _ModelConfig,
    cat_thresh: Dict[str, float],
    category_overlap_thresh: float,
    pick_indices: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """
    Replicates DoDetection for RTMDet network type.
    """
    orig_h, orig_w = image_rgb.shape[:2]

    blob, im_scale = _preprocess_centernet(
        image_rgb, cfg.img_scale, cfg.side_divisor, cfg.mean, cfg.std
    )

    outputs = sess.run(None, {sess.get_inputs()[0].name: blob})

    cls_scores = outputs[0:3]
    bbox_preds = outputs[3:6]
    strides = [8, 16, 32]

    scores, cls_idxs, dets = _decode_rtmdet(
        cls_scores,
        bbox_preds,
        strides=strides,
        orig_h=orig_h,
        orig_w=orig_w,
        im_scale=im_scale,
        score_thr=cfg.score_thr,
        max_per_im=cfg.max_per_im,
    )

    if len(scores) == 0:
        return []

    results_by_cat: Dict[int, List[Tuple[float, np.ndarray]]] = {}
    for i, (score, cls_idx) in enumerate(zip(scores, cls_idxs)):
        if pick_indices is not None and cls_idx not in pick_indices:
            continue
        cat_name = cfg.thresh_classes[cls_idx] if cls_idx < len(cfg.thresh_classes) else str(cls_idx)
        lo_thresh = cat_thresh.get(cat_name, cfg.score_thr)
        if score < lo_thresh:
            continue
        results_by_cat.setdefault(cls_idx, []).append((score, dets[i]))

    roi_list = []
    for cls_idx, items in results_by_cat.items():
        cat_name = cfg.thresh_classes[cls_idx] if cls_idx < len(cfg.thresh_classes) else str(cls_idx)
        cat_scores = np.array([s for s, _ in items])
        cat_dets = np.array([d for _, d in items])

        if category_overlap_thresh < 1.0:
            keep = _nms(cat_dets, cat_scores, category_overlap_thresh)
        else:
            keep = list(range(len(items)))

        for k in keep:
            roi_list.append({
                "name": cat_name,
                "score": float(cat_scores[k]),
                "box": cat_dets[k].tolist(),
            })

    return roi_list


def _object_passthrough(rois: List[Dict], color_scale: float) -> List[Dict]:
    return rois


def _recognize_object(rois: List[Dict], color_scale: float) -> List[Dict]:
    results = []
    for roi in rois:
        if not roi.get("detected", True):
            continue
        x1, y1, x2, y2 = roi["box"]
        if color_scale > 0:
            x1 = int(x1 / color_scale + 0.5)
            y1 = int(y1 / color_scale + 0.5)
            x2 = int(x2 / color_scale + 0.5)
            y2 = int(y2 / color_scale + 0.5)
        results.append({
            "name": roi["name"],
            "score": roi["score"],
            "box": [x1, y1, x2, y2],
        })
    return results


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class RTMDetDetect:
    """
    Local Python equivalent of the RTMDet product detection pipeline.
    """

    def __init__(
        self,
        model_dir: str,
        category_overlap_thresh: float = 0.3,
        centerness_sigma: Dict[str, float] = None,
        objectness_prior: Dict[str, float] = None,
        providers: Optional[List[str]] = None,
        edgebox_model_path: Optional[str] = str(
            Path(__file__).parent.parent.parent / "models" / "edgebox_model.yml.gz"
        ),
        disable_edgebox: bool = False,
        edgebox_num_detect: int = 1,
        threshold_file: str = "threshold.txt",
    ):
        model_dir = Path(model_dir)

        self._cfg = _ModelConfig(model_dir, threshold_file=threshold_file)

        self._cat_thresh: Dict[str, float] = dict(
            zip(self._cfg.thresh_classes, self._cfg.thresh_values)
        )

        self._category_overlap_thresh = category_overlap_thresh
        self._centerness_sigma: Dict[str, float] = centerness_sigma or {}
        self._objectness_prior: Dict[str, float] = objectness_prior or {}
        self._edgebox_model_path = edgebox_model_path
        self._disable_edgebox = disable_edgebox
        self._edgebox_num_detect = edgebox_num_detect

        onnx_path = str(model_dir / "model.onnx")
        self._sess = ort.InferenceSession(
            onnx_path,
            providers=_select_onnx_providers(providers),
        )

    def detect(
        self,
        image: np.ndarray,
        box_concept: str = "all",
        detection_limit: int = -1,
        multiple_object: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run the full detection pipeline on a single image.
        """
        color_img, color_scale = _image_color(image)

        validated = _validate_object_type(box_concept, self._cfg.thresh_classes)
        pick_indices = _pick_category_indices(validated, self._cfg.thresh_classes)

        rois = _roi_detect_product(
            color_img,
            self._sess,
            self._cfg,
            self._cat_thresh,
            self._category_overlap_thresh,
            pick_indices=pick_indices,
        )

        rois = _sort_fashion_objects(
            color_img, rois, self._centerness_sigma, self._objectness_prior
        )

        if not rois:
            fallback_name = _parse_output_type(validated)
            if fallback_name in (_CATEGORY_ALL, ""):
                fallback_name = _CATEGORY_OTHER

            if not self._disable_edgebox:
                fallback_rois = _detect_edgebox(
                    color_img, self._edgebox_model_path
                )
                if fallback_rois:
                    fallback_rois = _sort_fashion_objects(
                        color_img, fallback_rois, self._centerness_sigma, self._objectness_prior
                    )
                    fallback_rois = fallback_rois[: self._edgebox_num_detect]
                    for r in fallback_rois:
                        r["name"] = fallback_name
                    rois = fallback_rois
                else:
                    rois = _get_fashion_object_simple(color_img)
                    for r in rois:
                        r["name"] = fallback_name
            else:
                rois = _get_fashion_object_simple(color_img)
                for r in rois:
                    r["name"] = fallback_name

        rois = _object_passthrough(rois, color_scale)
        results = _recognize_object(rois, color_scale)

        if multiple_object and detection_limit >= 0 and len(results) > detection_limit:
            results = results[:detection_limit]

        return results
