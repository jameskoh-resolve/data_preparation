"""
SimpleDetect: lean CenterNet detector focused on throughput.

This module keeps only the essential detection path:
1) image resize/color normalization
2) ONNX inference (hm, hmax, wh)
3) decode + per-class threshold + NMS
4) rescale boxes to original image

Compared with centernet_detect.py, this version intentionally skips:
- objectness rescoring heuristics
- fallback detectors (EdgeBoxes / Canny fallback)
- category auto-expansion logic
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import yaml


def _image_color(image: np.ndarray, max_size: int = 1024) -> Tuple[np.ndarray, float]:
	"""Convert grayscale to 3-channel and cap max side to max_size."""
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


class _ModelConfig:
	"""Model metadata loaded from model.cfg/model_info.json/threshold.txt."""

	def __init__(self, model_dir: Path, threshold_file: str = "threshold.txt"):
		with open(model_dir / "model.cfg") as f:
			cfg = yaml.safe_load(f)

		self.mean = np.array(cfg["mean"], dtype=np.float32)
		self.std = np.array(cfg["std"], dtype=np.float32)
		self.img_scale: int = cfg.get("img_scale", 448)
		self.side_divisor: int = cfg.get("side_divisor", 32)
		self.down_ratio: int = cfg.get("down_ratio", 8)
		self.max_per_im: int = cfg.get("max_per_im", 100)
		self.score_thr: float = cfg.get("score_thr", 0.05)
		self.wh_agnostic: bool = cfg.get("wh_agnostic", True)

		with open(model_dir / "model_info.json") as f:
			info = json.load(f)
		self.input_blob: str = info.get("input_blob", "data")

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


def _preprocess_centernet(
	image_rgb: np.ndarray,
	cfg: _ModelConfig,
) -> Tuple[np.ndarray, Tuple[float, float]]:
	"""Resize + normalize + CHW + optional pad to side_divisor."""
	h, w = image_rgb.shape[:2]
	im_scale = (cfg.img_scale / h, cfg.img_scale / w)

	resized = cv2.resize(
		image_rgb,
		(cfg.img_scale, cfg.img_scale),
		interpolation=cv2.INTER_LINEAR,
	)

	img_f = resized.astype(np.float32) / 255.0
	img_f = (img_f - cfg.mean) / cfg.std
	img_chw = img_f.transpose(2, 0, 1)

	_, rh, rw = img_chw.shape
	pad_h = math.ceil(rh / cfg.side_divisor) * cfg.side_divisor
	pad_w = math.ceil(rw / cfg.side_divisor) * cfg.side_divisor

	blob = np.zeros((1, 3, pad_h, pad_w), dtype=np.float32)
	blob[0, :, :rh, :rw] = img_chw

	return blob, im_scale


def _decode_centernet(
	hm: np.ndarray,
	hmax: np.ndarray,
	wh: np.ndarray,
	im_scale: Tuple[float, float],
	orig_h: int,
	orig_w: int,
	score_thr: float,
	max_per_im: int,
	down_ratio: int,
	wh_agnostic: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Decode CenterNet hm/hmax/wh outputs into (scores, class_ids, boxes)."""
	_, num_classes, map_h, map_w = hm.shape
	area = map_h * map_w

	hm_flat = hm[0].reshape(num_classes, area)
	hmax_flat = hmax[0].reshape(num_classes, area)
	wh_flat = wh[0].reshape(-1, area)

	mask = (hmax_flat == hm_flat) & (hmax_flat >= score_thr)
	indices = np.argwhere(mask)
	if len(indices) == 0:
		return np.array([]), np.array([]), np.zeros((0, 4))

	cls_idx = indices[:, 0]
	sp_idx = indices[:, 1]
	scores = hm_flat[cls_idx, sp_idx]

	order = np.argsort(-scores)[:max_per_im]
	cls_idx = cls_idx[order]
	sp_idx = sp_idx[order]
	scores = scores[order]

	xs = (sp_idx % map_w) * down_ratio
	ys = (sp_idx // map_w) * down_ratio

	if wh_agnostic:
		wh_cls_offset = np.zeros(len(cls_idx), dtype=int)
	else:
		wh_cls_offset = 4 * cls_idx

	x1 = np.clip((xs - wh_flat[wh_cls_offset + 0, sp_idx]) / im_scale[1], 0, orig_w - 1)
	y1 = np.clip((ys - wh_flat[wh_cls_offset + 1, sp_idx]) / im_scale[0], 0, orig_h - 1)
	x2 = np.clip((xs + wh_flat[wh_cls_offset + 2, sp_idx]) / im_scale[1], 0, orig_w - 1)
	y2 = np.clip((ys + wh_flat[wh_cls_offset + 3, sp_idx]) / im_scale[0], 0, orig_h - 1)

	dets = np.stack([x1, y1, x2, y2], axis=1)
	return scores, cls_idx, dets


def _nms(dets: np.ndarray, scores: np.ndarray, iou_thresh: float) -> List[int]:
	"""Standard greedy NMS. Returns indices kept in descending score order."""
	if len(dets) == 0:
		return []

	x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)
	order = np.argsort(-scores)
	keep: List[int] = []

	while len(order) > 0:
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


def _parse_box_concept(box_concept: str, class_names: List[str]) -> Optional[Set[int]]:
	"""Parse a ';'-separated class filter. Returns None to keep all classes."""
	if not box_concept or box_concept in ("all", "other", "wholeimage"):
		return None

	name_to_idx = {name: i for i, name in enumerate(class_names)}
	out: Set[int] = set()
	for token in box_concept.split(";"):
		token = token.strip()
		if token in name_to_idx:
			out.add(name_to_idx[token])
	return out if out else None


def _rescale_to_original(rois: List[Dict[str, Any]], color_scale: float) -> List[Dict[str, Any]]:
	"""Map boxes from resized-image coordinates back to original image space."""
	if color_scale <= 0:
		return rois

	results: List[Dict[str, Any]] = []
	for roi in rois:
		x1, y1, x2, y2 = roi["box"]
		results.append(
			{
				"name": roi["name"],
				"score": roi["score"],
				"box": [
					int(x1 / color_scale + 0.5),
					int(y1 / color_scale + 0.5),
					int(x2 / color_scale + 0.5),
					int(y2 / color_scale + 0.5),
				],
			}
		)
	return results


class SimpleDetect:
	"""
	Fast, minimal CenterNet detector for bulk inference.

	It is API-compatible with ProductDetect.detect for the common arguments,
	but intentionally omits objectness rescoring and fallback detection logic.
	"""

	def __init__(
		self,
		model_dir: str,
		category_overlap_thresh: float = 0.3,
		providers: Optional[List[str]] = None,
		threshold_file: str = "threshold.txt",
	):
		model_path = Path(model_dir)
		self._cfg = _ModelConfig(model_path, threshold_file=threshold_file)
		self._category_overlap_thresh = category_overlap_thresh
		self._cat_thresh: Dict[str, float] = dict(
			zip(self._cfg.thresh_classes, self._cfg.thresh_values)
		)

		onnx_path = str(model_path / "model.onnx")
		self._sess = ort.InferenceSession(
			onnx_path,
			providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
		)

	def detect(
		self,
		image: np.ndarray,
		box_concept: str = "all",
		detection_limit: int = -1,
		multiple_object: bool = False,
	) -> List[Dict[str, Any]]:
		"""
		Run simplified CenterNet detection.

		Returns detections as:
			[{"name": str, "score": float, "box": [x1, y1, x2, y2]}, ...]
		"""
		color_img, color_scale = _image_color(image)
		orig_h, orig_w = color_img.shape[:2]

		pick_indices = _parse_box_concept(box_concept, self._cfg.thresh_classes)

		blob, im_scale = _preprocess_centernet(color_img, self._cfg)
		input_name = self._sess.get_inputs()[0].name
		outputs = self._sess.run(None, {input_name: blob})

		hm, hmax, wh = outputs[0], outputs[1], outputs[2]
		scores, cls_idxs, dets = _decode_centernet(
			hm,
			hmax,
			wh,
			im_scale=im_scale,
			orig_h=orig_h,
			orig_w=orig_w,
			score_thr=self._cfg.score_thr,
			max_per_im=self._cfg.max_per_im,
			down_ratio=self._cfg.down_ratio,
			wh_agnostic=self._cfg.wh_agnostic,
		)

		if len(scores) == 0:
			return []

		by_cat: Dict[int, List[Tuple[float, np.ndarray]]] = {}
		for i, (score, cls_idx) in enumerate(zip(scores, cls_idxs)):
			if pick_indices is not None and cls_idx not in pick_indices:
				continue
			cat_name = (
				self._cfg.thresh_classes[cls_idx]
				if cls_idx < len(self._cfg.thresh_classes)
				else str(cls_idx)
			)
			if score < self._cat_thresh.get(cat_name, self._cfg.score_thr):
				continue
			by_cat.setdefault(cls_idx, []).append((float(score), dets[i]))

		rois: List[Dict[str, Any]] = []
		for cls_idx, items in by_cat.items():
			cat_name = (
				self._cfg.thresh_classes[cls_idx]
				if cls_idx < len(self._cfg.thresh_classes)
				else str(cls_idx)
			)
			cat_scores = np.array([s for s, _ in items], dtype=np.float32)
			cat_dets = np.array([d for _, d in items], dtype=np.float32)

			if self._category_overlap_thresh < 1.0:
				keep = _nms(cat_dets, cat_scores, self._category_overlap_thresh)
			else:
				keep = list(range(len(items)))

			for k in keep:
				rois.append(
					{
						"name": cat_name,
						"score": float(cat_scores[k]),
						"box": cat_dets[k].tolist(),
					}
				)

		if not rois:
			return []

		rois.sort(key=lambda x: -x["score"])
		results = _rescale_to_original(rois, color_scale)

		if multiple_object and detection_limit >= 0 and len(results) > detection_limit:
			results = results[:detection_limit]

		return results

