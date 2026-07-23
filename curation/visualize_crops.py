#!/usr/bin/env python
"""Standalone LLM crop visualizer.

Reads processed_cache.json + llm_cache.json from the cache directory,
re-generates the exact crops sent to the LLM (using the same padding config),
and writes a self-contained HTML gallery.

Usage:
    PYTHONPATH=. .venv/bin/python curation/visualize_crops.py configs/auto_annotate_westside_part_2.yaml
    PYTHONPATH=. .venv/bin/python curation/visualize_crops.py configs/auto_annotate_westside_part_2.yaml --output my_review.html
"""

from __future__ import annotations

import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import typer
from loguru import logger
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from curation.auto_annotate import (
    PredictionCache,
    _stable_image_id,
    _upscale_crop_if_too_small,
    load_image_and_path,
    prepare_crop,
    resolve_dataset_csv,
)

app = typer.Typer(pretty_exceptions_show_locals=False)


def build_crop_html(
    cfg_path: str,
    output_filename: str = "crops.html",
) -> Path:
    cfg = OmegaConf.load(str(cfg_path))
    dataset_cfg = cfg.get("dataset", {})
    llm_cfg_root = cfg.get("llm_validation", cfg.get("llm validation", {}))

    output_dir = Path(dataset_cfg.get("output_dir", REPO_ROOT / "curated_datasets/curation"))
    custom_cache = dataset_cfg.get("cache_dir")
    cache_dir = Path(custom_cache) if custom_cache else output_dir / "cache"
    image_cache_dir = cache_dir / "image_cache"

    processed_cache = PredictionCache(cache_dir / "processed_cache.json")
    llm_cache       = PredictionCache(cache_dir / "llm_cache.json")

    # Compute processed_cfg_hash (must match the hash used during the main run)
    detection_models_raw = cfg.get("detection_models", cfg.get("detection models", []))
    if isinstance(detection_models_raw, dict):
        detection_models_raw = (
            [detection_models_raw] if "model_type" in detection_models_raw
            else list(detection_models_raw.values())
        )
    detection_models_list = [
        OmegaConf.to_container(m, resolve=True) if hasattr(m, "_metadata") else dict(m)
        for m in detection_models_raw
    ]
    dedup_cfg_raw = cfg.get("dedup_policy", cfg.get("dedup policy", {}))
    if dedup_cfg_raw:
        dedup_cfg_raw = OmegaConf.to_container(dedup_cfg_raw, resolve=True)
    keep_bbs = bool(dataset_cfg.get("keep_bounding_boxes", False))
    processed_cfg_hash = hashlib.md5(
        json.dumps(
            {"detection_models": detection_models_list, "dedup_policy": dedup_cfg_raw, "keep_bounding_boxes": keep_bbs},
            sort_keys=True, default=str,
        ).encode()
    ).hexdigest()[:8]

    llm_classes = list(llm_cfg_root.get("classes", []))

    # im_ids that have processed cache entries
    im_ids_with_processed = set()
    for key in processed_cache.cache:
        if key.startswith("processed:"):
            parts = key.split(":")
            if len(parts) >= 3:
                im_ids_with_processed.add(parts[1])

    # Build im_id -> im_url from CSV (local path only, no azure)
    df, _ = resolve_dataset_csv(dataset_cfg, prefer_azure=False)
    image_col = next(
        (c for c in ("im_url", "image_url", "url", "image", "original_url") if c in df.columns), None
    )
    if not image_col:
        raise ValueError("Could not find image URL column in dataset.")
    im_id_to_url = {
        row.get("im_id", _stable_image_id(str(row[image_col]).strip())): str(row[image_col]).strip()
        for _, row in df.iterrows()
    }

    crop_entries = []
    total_images = 0

    for im_id in sorted(im_ids_with_processed):
        cached_processed = processed_cache.get(f"processed:{im_id}:{processed_cfg_hash}")
        if not cached_processed:
            continue

        llm_dets = (
            [d for d in cached_processed if d.get("name") in llm_classes]
            if llm_classes else list(cached_processed)
        )
        if not llm_dets:
            continue

        im_url = im_id_to_url.get(im_id, "")
        image = None
        local_path = image_cache_dir / f"{im_id}.jpg"
        if local_path.exists():
            image = cv2.imread(str(local_path))
        if image is None and im_url:
            image, _ = load_image_and_path(im_url, image_cache_dir)
        if image is None:
            logger.warning("Could not load image for im_id {}", im_id)
            continue

        total_images += 1

        for det in llm_dets:
            class_name = det.get("name", "")
            box = det.get("box", [0, 0, 0, 0])
            box_str = f"[{int(round(box[0]))},{int(round(box[1]))},{int(round(box[2]))},{int(round(box[3]))}]"

            matched_llm = next(
                (v for k, v in llm_cache.cache.items()
                 if k.startswith(f"llm:{im_id}:{class_name}:{box_str}:") and isinstance(v, dict)),
                None,
            )
            if matched_llm is None:
                continue

            class_override = llm_cfg_root.get(class_name, {})
            general_cfg = llm_cfg_root.get("general", {})

            def get_cfg_val(key, default=None):
                return class_override.get(key, general_cfg.get(key, default))

            padding = float(get_cfg_val("padding", 0.0))
            fp = get_cfg_val("flat_padding_px", None)
            mbp = get_cfg_val("min_box_padding", None)
            flat_padding_px = int(fp) if fp is not None else None
            min_box_padding = int(mbp) if mbp is not None else None
            min_crop_short_side = int(get_cfg_val("min_crop_short_side", 192))
            min_crop_long_side = int(get_cfg_val("min_crop_long_side", 256))
            jpeg_quality = max(1, min(100, int(get_cfg_val("jpeg_quality", 98))))
            interpolation_map = {
                "nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
                "area": cv2.INTER_AREA, "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4,
            }
            interpolation = interpolation_map.get(
                str(get_cfg_val("upscale_interpolation", "cubic")).strip().lower(),
                cv2.INTER_CUBIC,
            )

            crop = prepare_crop(image, box, padding=padding, flat_padding_px=flat_padding_px, min_box_padding=min_box_padding)
            if crop.size == 0:
                continue
            crop = _upscale_crop_if_too_small(crop, min_short_side=min_crop_short_side, min_long_side=min_crop_long_side, interpolation=interpolation)

            success, encoded_img = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
            if not success:
                continue

            crop_entries.append({
                "im_id": im_id,
                "im_url": im_url,
                "class_name": class_name,
                "box": box,
                "is_valid": bool(matched_llm.get("is_valid")),
                "reason": str(matched_llm.get("reason", "") or ""),
                "crop_b64": base64.b64encode(encoded_img.tobytes()).decode("ascii"),
            })

    total_crops = len(crop_entries)
    total_images = len({e["im_id"] for e in crop_entries})
    logger.info("Collected {} crops from {} images", total_crops, total_images)

    html_out = output_dir / output_filename
    html_out.write_text(_render_html(crop_entries))
    logger.info("Saved crop visualization ({} crops) to {}", total_crops, html_out)
    return html_out


def _render_html(crop_entries: list) -> str:
    cards_html = []
    for e in crop_entries:
        border_color = "#22c55e" if e["is_valid"] else "#ef4444"
        badge_bg = "#166534" if e["is_valid"] else "#7f1d1d"
        badge_text = "✓ valid" if e["is_valid"] else "✗ invalid"
        reason_html = f'<div class="reason">{e["reason"]}</div>' if e["reason"] else ""
        cards_html.append(f"""
        <div class="card {'valid' if e['is_valid'] else 'invalid'}" style="border-color:{border_color}">
          <img src="data:image/jpeg;base64,{e['crop_b64']}" alt="{e['class_name']}">
          <div class="card-info">
            <span class="badge" style="background:{badge_bg}">{badge_text}</span>
            <span class="cls">{e['class_name']}</span>
            {reason_html}
            <div class="imid">{e['im_id']}</div>
          </div>
        </div>""")

    all_classes = sorted({e["class_name"] for e in crop_entries})
    filter_buttons = "".join(
        f'<button class="filter-btn active" data-cls="{c}" onclick="toggleFilter(this)">{c}</button>'
        for c in all_classes
    )
    total_crops = len(crop_entries)
    total_images = len({e["im_id"] for e in crop_entries})
    valid_count = sum(1 for e in crop_entries if e["is_valid"])
    invalid_count = total_crops - valid_count

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLM Crop Viewer</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ background:#0f172a; color:#e2e8f0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; min-height:100vh; }}
.header {{ position:sticky; top:0; z-index:10; background:#1e293b; border-bottom:1px solid #334155; padding:12px 20px; display:flex; align-items:center; gap:12px; flex-wrap:wrap; }}
.header h1 {{ font-size:15px; font-weight:700; color:#f8fafc; }}
.stat {{ font-size:11px; background:#0f172a; border:1px solid #334155; border-radius:12px; padding:3px 10px; color:#94a3b8; }}
.filter-btn {{ font-size:11px; padding:3px 10px; border-radius:12px; border:1px solid #334155; background:#1e293b; color:#94a3b8; cursor:pointer; transition:all .15s; }}
.filter-btn.active {{ background:#3b82f6; border-color:#3b82f6; color:#fff; }}
.validity-btn {{ font-size:11px; padding:3px 10px; border-radius:12px; cursor:pointer; transition:all .15s; }}
#btn-valid {{ border:1px solid #22c55e; background:#14532d; color:#86efac; }}
#btn-invalid {{ border:1px solid #ef4444; background:#7f1d1d; color:#fca5a5; }}
#btn-valid.inactive, #btn-invalid.inactive {{ background:#1e293b; border-color:#334155; color:#64748b; }}
.grid {{ display:flex; flex-wrap:wrap; gap:12px; padding:16px; }}
.card {{ border:2px solid; border-radius:8px; overflow:hidden; background:#1e293b; width:200px; display:flex; flex-direction:column; transition:transform .15s; }}
.card:hover {{ transform:scale(1.03); }}
.card img {{ width:100%; height:180px; object-fit:contain; background:#0f172a; }}
.card-info {{ padding:8px; font-size:11px; display:flex; flex-direction:column; gap:4px; }}
.badge {{ display:inline-block; padding:2px 7px; border-radius:4px; font-size:10px; font-weight:700; color:#fff; width:fit-content; }}
.cls {{ font-weight:700; font-size:12px; color:#f1f5f9; }}
.reason {{ color:#94a3b8; font-size:10px; line-height:1.4; }}
.imid {{ color:#475569; font-size:9px; font-family:monospace; margin-top:2px; word-break:break-all; }}
</style>
</head>
<body>
<div class="header">
  <h1>LLM Crop Viewer</h1>
  <span class="stat">{total_crops} crops / {total_images} images</span>
  <span class="stat" style="color:#86efac">{valid_count} valid</span>
  <span class="stat" style="color:#fca5a5">{invalid_count} invalid</span>
  <button id="btn-valid" class="validity-btn" onclick="toggleValidity('valid')">show valid</button>
  <button id="btn-invalid" class="validity-btn" onclick="toggleValidity('invalid')">show invalid</button>
  {filter_buttons}
</div>
<div class="grid" id="grid">
{''.join(cards_html)}
</div>
<script>
const showValid = {{v:true}}, showInvalid = {{v:true}};
const activeClasses = new Set({json.dumps(all_classes)});
function applyFilters() {{
  document.querySelectorAll('.card').forEach(c => {{
    const isValid = c.classList.contains('valid');
    const cls = c.querySelector('.cls').textContent;
    const show = activeClasses.has(cls) && ((isValid && showValid.v) || (!isValid && showInvalid.v));
    c.style.display = show ? '' : 'none';
  }});
}}
function toggleFilter(btn) {{
  const cls = btn.dataset.cls;
  if (activeClasses.has(cls)) {{ activeClasses.delete(cls); btn.classList.remove('active'); }}
  else {{ activeClasses.add(cls); btn.classList.add('active'); }}
  applyFilters();
}}
function toggleValidity(which) {{
  if (which === 'valid') {{ showValid.v = !showValid.v; document.getElementById('btn-valid').classList.toggle('inactive', !showValid.v); }}
  else {{ showInvalid.v = !showInvalid.v; document.getElementById('btn-invalid').classList.toggle('inactive', !showInvalid.v); }}
  applyFilters();
}}
</script>
</body>
</html>"""


@app.command()
def main(
    config_file: str = typer.Argument(..., help="Path to auto-annotate config YAML"),
    output: str = typer.Option("crops.html", help="Output HTML filename (relative to output_dir)"),
):
    """Generate HTML gallery of crops sent to the LLM, with validation results."""
    build_crop_html(config_file, output)


if __name__ == "__main__":
    app()
