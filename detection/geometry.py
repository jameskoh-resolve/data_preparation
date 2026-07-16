"""Geometry and suppression helpers for detection evaluation."""

from typing import Any, Callable, Dict, List, Optional


def compute_iou(box_a, box_b) -> float:
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
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
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def box_area_ratio(box, image_h: int, image_w: int) -> float:
    """Compute area(box) / area(image)."""
    x1, y1, x2, y2 = box
    box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    image_area = float(image_h * image_w)
    return box_area / image_area if image_area > 0 else 0.0


def same_class_nms(detections: list, map_prediction: Callable[[str], Optional[str]], iou_thresh: float) -> list:
    """Suppress duplicate detections within the same mapped GT class."""
    by_class: Dict[str, list] = {}
    no_class = []
    for det in detections:
        mapped = map_prediction(str(det["name"]))
        if mapped is None:
            no_class.append(det)
        else:
            by_class.setdefault(mapped, []).append(det)

    kept = list(no_class)
    for _, dets in by_class.items():
        dets_sorted = sorted(dets, key=lambda d: float(d["score"]), reverse=True)
        suppressed = [False] * len(dets_sorted)
        for i in range(len(dets_sorted)):
            if suppressed[i]:
                continue
            for j in range(i + 1, len(dets_sorted)):
                if suppressed[j]:
                    continue
                iou = compute_iou(dets_sorted[i]["box"], dets_sorted[j]["box"])
                if iou >= iou_thresh:
                    suppressed[j] = True
        kept.extend(d for d, s in zip(dets_sorted, suppressed) if not s)
    return kept


def cross_class_nms(
    items: list,
    suppression_rules: Dict[str, List[str]],
    iou_thresh: float,
    class_key: str = "box_concept",
) -> list:
    """Suppress boxes of generic classes when a more specific class overlaps."""
    if not items:
        return items

    suppressed = set()
    for i, item_a in enumerate(items):
        cls_a = item_a.get(class_key, "")
        suppressible = suppression_rules.get(cls_a)
        if not suppressible:
            continue
        box_a = item_a.get("box") if item_a.get("box") is not None else item_a.get("box_list")
        if box_a is None:
            continue
        for j, item_b in enumerate(items):
            if i == j or j in suppressed:
                continue
            cls_b = item_b.get(class_key, "")
            if cls_b not in suppressible:
                continue
            box_b = item_b.get("box") if item_b.get("box") is not None else item_b.get("box_list")
            if box_b is None:
                continue
            if compute_iou(box_a, box_b) >= iou_thresh:
                suppressed.add(j)

    return [item for idx, item in enumerate(items) if idx not in suppressed]


def gt_nms(gt_items: list, iou_thresh: float) -> list:
    """Suppress overlapping GT boxes within the same concept class."""
    by_concept: Dict[str, list] = {}
    for item in gt_items:
        by_concept.setdefault(item["box_concept"], []).append(item)

    kept = []
    for _, items in by_concept.items():
        items_sorted = sorted(
            items,
            key=lambda it: (it["box"][2] - it["box"][0]) * (it["box"][3] - it["box"][1]),
            reverse=True,
        )
        suppressed = [False] * len(items_sorted)
        for i in range(len(items_sorted)):
            if suppressed[i]:
                continue
            for j in range(i + 1, len(items_sorted)):
                if suppressed[j]:
                    continue
                iou = compute_iou(items_sorted[i]["box"], items_sorted[j]["box"])
                if iou >= iou_thresh:
                    suppressed[j] = True
        kept.extend(it for it, s in zip(items_sorted, suppressed) if not s)
    return kept
