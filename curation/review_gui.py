#!/usr/bin/env python
"""Interactive annotation review GUI.

A local HTTP server + browser frontend for reviewing, correcting, and extending
detections from the auto-annotate pipeline.

Features
--------
- Grid (batch) view: shows as many image thumbnails at once as fit your screen
  resolution, with bounding boxes drawn on each thumbnail:
    green = valid       red    = invalid (LLM rejected the detection)
    violet = new (human-added)
  Invalid boxes are hidden by default (on both the grid thumbnails and the
  detail view) so you can quickly scan a batch for detections that need
  attention. Click a thumbnail to open the detail view for that image.
- Keyboard hotkeys:
    A / ←   prev batch (grid view) / prev image (detail view)
    D / →   next batch (grid view) / next image (detail view)
    Space   show/hide ALL invalid boxes at once (grid + detail view)
    Enter   approve selected box (detail view only)
    X       reject selected box (detail view only)
    Esc     back to grid view (or cancel draw mode if active)
  Individual boxes are shown/hidden by clicking their Hide/Show button in the
  detail view's box list (not via keyboard).
- Click-drag on canvas (detail view) to draw a new box (select class first in
  the right panel)
- Save decisions to ``<cache_dir>/human_review.json``
- Saving in the GUI does NOT by itself change any dataset CSV. Run the
  ``export`` command below to apply your decisions and produce a final
  ``{dataset}_human_reviewed.csv``.

Usage
-----
    PYTHONPATH=. .venv/bin/python curation/review_gui.py main configs/auto_annotate_westside_part_2.yaml
    PYTHONPATH=. .venv/bin/python curation/review_gui.py main configs/auto_annotate_westside_part_2.yaml --port 7654
    PYTHONPATH=. .venv/bin/python curation/review_gui.py export configs/auto_annotate_westside_part_2.yaml
"""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Optional

import pandas as pd
import typer
from loguru import logger
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from curation.auto_annotate import (
    PredictionCache,
    _stable_image_id,
    load_image_and_path,
    resolve_dataset_csv,
)

app = typer.Typer(pretty_exceptions_show_locals=False)

CLASSES_FILE = REPO_ROOT / "product_item_13_04" / "classes.txt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_classes() -> list[str]:
    if CLASSES_FILE.exists():
        return [ln.strip() for ln in CLASSES_FILE.read_text().splitlines() if ln.strip()]
    return []


def _processed_cfg_hash(cfg) -> str:
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
    keep_bbs = bool(cfg.get("dataset", {}).get("keep_bounding_boxes", False))
    return hashlib.md5(
        json.dumps(
            {"detection_models": detection_models_list, "dedup_policy": dedup_cfg_raw, "keep_bounding_boxes": keep_bbs},
            sort_keys=True, default=str,
        ).encode()
    ).hexdigest()[:8]


def load_pipeline_data(cfg_path: str) -> tuple[list[dict], Path, Path]:
    """Load all pipeline cache data.

    Returns
    -------
    images_data : list of {im_id, im_url, boxes}
    image_cache_dir : directory where cached images live
    review_path : path to human_review.json
    """
    cfg = OmegaConf.load(str(cfg_path))
    dataset_cfg = cfg.get("dataset", {})
    llm_cfg_root = cfg.get("llm_validation", cfg.get("llm validation", {}))

    output_dir = Path(dataset_cfg.get("output_dir", REPO_ROOT / "curated_datasets/curation"))
    custom_cache = dataset_cfg.get("cache_dir")
    cache_dir = Path(custom_cache) if custom_cache else output_dir / "cache"
    image_cache_dir = cache_dir / "image_cache"

    processed_cache = PredictionCache(cache_dir / "processed_cache.json")
    llm_cache = PredictionCache(cache_dir / "llm_cache.json")
    review_path = cache_dir / "human_review.json"

    cfg_hash = _processed_cfg_hash(cfg)
    llm_classes: list[str] = list(llm_cfg_root.get("classes", []))

    df, _ = resolve_dataset_csv(dataset_cfg, prefer_azure=False)
    image_col = next(
        (c for c in ("im_url", "image_url", "url", "image", "original_url") if c in df.columns), None
    )
    if not image_col:
        raise ValueError("Could not find image URL column in dataset.")

    images_data: list[dict] = []

    for _, row in df.iterrows():
        im_url = str(row[image_col]).strip()
        if not im_url:
            continue
        im_id = str(row.get("im_id", _stable_image_id(im_url)))
        # The image cache always uses the MD5 of the URL, regardless of im_id
        img_cache_key = _stable_image_id(im_url)

        cached_processed = processed_cache.get(f"processed:{im_id}:{cfg_hash}")
        if not cached_processed:
            continue

        boxes: list[dict] = []
        for det in cached_processed:
            box = dict(det)
            coords = box.get("box", [0, 0, 0, 0])
            box_str = f"[{int(round(coords[0]))},{int(round(coords[1]))},{int(round(coords[2]))},{int(round(coords[3]))}]"
            cls_name = box.get("name", "")

            matched_llm = next(
                (v for k, v in llm_cache.cache.items()
                 if k.startswith(f"llm:{im_id}:{cls_name}:{box_str}:") and isinstance(v, dict)),
                None,
            )
            if matched_llm is not None:
                box["llm_validated"] = True
                box["is_valid"] = bool(matched_llm.get("is_valid", True))
                box["reason"] = str(matched_llm.get("reason", "") or "")
            else:
                box["llm_validated"] = False
                box["is_valid"] = True
                box["reason"] = ""

            boxes.append(box)

        images_data.append({
            "im_id": im_id,
            "im_url": im_url,
            "_img_cache_key": img_cache_key,  # server-side only, stripped before sending to browser
            "boxes": boxes,
        })

    logger.info("Loaded {} images from pipeline cache", len(images_data))
    return images_data, image_cache_dir, review_path


def _box_uid(im_id: str, name: str, box: list) -> str:
    """Matches the frontend's boxUid() exactly, so decisions keyed by the browser
    line up with boxes reloaded straight from the pipeline cache."""
    coords = box or [0, 0, 0, 0]
    x1, y1, x2, y2 = (int(round(float(v))) for v in coords)
    return f"{im_id}:{name}:{x1}:{y1}:{x2}:{y2}"


def build_human_reviewed_csv(cfg_path: str) -> Path:
    """Apply saved human review decisions on top of the pipeline cache and write
    a new ``{dataset}_human_reviewed.csv``, without touching the original
    ``{dataset}_annotated.csv``.

    Decision rule per box (matches the review GUI's semantics):
      - explicit human 'rejected' -> excluded
      - explicit human 'approved' -> included (overrides the LLM verdict)
      - no explicit decision      -> included iff the LLM (or lack of LLM
                                     validation) already marked it valid
    Human-added boxes (drawn in the GUI) are always included. Note that the
    'hidden' flag is purely a review-UI visibility toggle and does NOT affect
    inclusion here.
    """
    images_data, _, review_path = load_pipeline_data(cfg_path)
    images_by_id = {img["im_id"]: img for img in images_data}

    cfg = OmegaConf.load(str(cfg_path))
    dataset_cfg = cfg.get("dataset", {})
    df, _ = resolve_dataset_csv(dataset_cfg, prefer_azure=False)
    image_col = next(
        (c for c in ("im_url", "image_url", "url", "image", "original_url") if c in df.columns), None
    )
    if not image_col:
        raise ValueError("Could not find image URL column in dataset.")

    human_review: dict[str, Any] = {"decisions": {}, "added_boxes": {}, "hidden": {}}
    if review_path.exists():
        try:
            human_review = json.loads(review_path.read_text())
        except Exception as exc:
            logger.warning("Could not load existing human_review.json: {}", exc)
    decisions: dict[str, str] = human_review.get("decisions", {}) or {}
    added_boxes: dict[str, list] = human_review.get("added_boxes", {}) or {}

    n_rejected = n_approved_override = n_added = 0
    output_rows = []
    for _, row in df.iterrows():
        im_url = str(row[image_col]).strip()
        im_id = str(row.get("im_id", _stable_image_id(im_url))) if im_url else ""
        img_entry = images_by_id.get(im_id)

        final_dets: list[dict] = []
        if img_entry is not None:
            for box in img_entry["boxes"]:
                uid = _box_uid(im_id, box.get("name", ""), box.get("box", [0, 0, 0, 0]))
                decision = decisions.get(uid)
                if decision == "rejected":
                    n_rejected += 1
                    continue
                if decision == "approved":
                    n_approved_override += 1
                    final_dets.append(box)
                    continue
                if box.get("is_valid", True):
                    final_dets.append(box)

        for added in added_boxes.get(im_id, []):
            final_dets.append({"name": added.get("name", ""), "box": added.get("box", [0, 0, 0, 0])})
            n_added += 1

        concepts_str = ",".join(d.get("name", "") for d in final_dets) if final_dets else None
        boxes_str = (
            ",".join(
                f"[{int(round(b[0]))},{int(round(b[1]))},{int(round(b[2]))},{int(round(b[3]))}]"
                for b in (d.get("box", [0, 0, 0, 0]) for d in final_dets)
            ) if final_dets else None
        )

        row_updated = dict(row)
        row_updated["concepts"] = concepts_str
        row_updated["boxes"] = boxes_str
        output_rows.append(row_updated)

    out_df = pd.DataFrame(output_rows)
    if str(dataset_cfg.get("type", "flat csv")).lower() == "hydravision":
        output_stem = dataset_cfg.get("dataset_name", "annotated_dataset")
    else:
        output_stem = Path(dataset_cfg.get("path", "annotated_dataset")).stem
    output_dir = Path(dataset_cfg.get("output_dir", REPO_ROOT / "curated_datasets/curation"))
    out_path = output_dir / f"{output_stem}_human_reviewed.csv"
    out_df.to_csv(out_path, index=False)
    logger.info(
        "Human-reviewed CSV written to {} ({} rows). {} boxes rejected by human, "
        "{} approved-override, {} human-added boxes included.",
        out_path, len(out_df), n_rejected, n_approved_override, n_added,
    )
    return out_path


# ---------------------------------------------------------------------------
# Frontend HTML (single-page app, fully embedded)
# ---------------------------------------------------------------------------

_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Annotation Review</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body { height: 100%; overflow: hidden; background: #0f172a; color: #e2e8f0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; font-size: 13px; }

/* ── Layout ────────────────────────────────────────────────────────────── */
.app { display: flex; flex-direction: column; height: 100vh; }
.toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  background: #1e293b; border-bottom: 1px solid #334155; padding: 7px 12px; flex-shrink: 0; }
.main-row { display: flex; flex: 1; overflow: hidden; }
.nav-group { display: flex; align-items: center; gap: 8px; }

/* ── Grid view (batches of thumbnails) ───────────────────────────────────── */
.view { flex: 1; display: flex; overflow: hidden; min-height: 0; }
#gridView { flex-direction: column; }
#detailView { flex-direction: row; }
.grid-wrap { flex: 1; overflow: hidden; padding: 10px; min-height: 0; }
.grid { display: grid; gap: 10px; height: 100%; width: 100%; }
.tile { position: relative; background: #0f172a; border: 2px solid #1e293b; border-radius: 6px;
  overflow: hidden; display: flex; flex-direction: column; cursor: pointer; }
.tile:hover { border-color: #3b82f6; }
.tile canvas { flex: 1; display: block; width: 100%; min-height: 0; }
.tile-label { flex-shrink: 0; height: 20px; padding: 2px 6px; font-size: 10px; color: #94a3b8;
  background: #1a2438; display: flex; justify-content: space-between; align-items: center;
  gap: 4px; white-space: nowrap; overflow: hidden; }
.tile-label .t-id { overflow: hidden; text-overflow: ellipsis; font-family: monospace; }
.tile-badge { background: #7f1d1d; color: #fca5a5; font-size: 9px; font-weight: 700;
  padding: 0 4px; border-radius: 3px; flex-shrink: 0; }

/* ── Canvas (detail view centre) ──────────────────────────────────────────── */
.canvas-panel { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
.canvas-wrapper { flex: 1; position: relative; overflow: hidden; min-height: 0; }
#canvas { position: absolute; inset: 0; cursor: default; background: #0a0f1e; }
.info-bar { flex-shrink: 0; background: #1e293b; border-top: 1px solid #334155;
  padding: 5px 12px; font-size: 10px; color: #475569; display: flex; gap: 16px; align-items: center; }
#draw-hint { color: #f59e0b; font-weight: 600; }

/* ── Box panel (right) ─────────────────────────────────────────────────── */
.box-panel { width: 230px; flex-shrink: 0; display: flex; flex-direction: column;
  background: #1a2438; border-left: 1px solid #334155; overflow: hidden; }
.box-panel-hdr { padding: 6px 10px; font-size: 11px; font-weight: 600; color: #94a3b8;
  border-bottom: 1px solid #334155; flex-shrink: 0; display: flex; justify-content: space-between; }
.cls-section { padding: 7px 8px; border-bottom: 1px solid #334155; flex-shrink: 0; }
.cls-section label { font-size: 10px; color: #64748b; display: block; margin-bottom: 3px; }
select { width: 100%; background: #0f172a; border: 1px solid #334155; color: #e2e8f0;
  border-radius: 4px; padding: 4px 6px; font-size: 11px; outline: none; }
select:focus { border-color: #3b82f6; }
.box-scroll { flex: 1; overflow-y: auto; padding: 3px; display: flex; flex-direction: column; gap: 2px; }

/* ── Box cards ─────────────────────────────────────────────────────────── */
.box-card { padding: 6px 8px; border-radius: 5px; cursor: pointer;
  border: 1px solid #1e293b; background: #0f172a; transition: border-color 0.1s; }
.box-card:hover { border-color: #334155; }
.box-card.selected { border-color: #38bdf8; background: #0c2340; }
.box-card-top { display: flex; justify-content: space-between; align-items: center; gap: 4px; }
.box-name { font-weight: 700; font-size: 12px; }
.box-badge { font-size: 9px; font-weight: 700; padding: 1px 5px; border-radius: 3px; white-space: nowrap; }
.b-valid    { background: #14532d; color: #86efac; }
.b-invalid  { background: #7f1d1d; color: #fca5a5; }
.b-approved { background: #064e3b; color: #34d399; }
.b-rejected { background: #450a0a; color: #f87171; }
.b-new      { background: #4c1d95; color: #ddd6fe; }
.b-hidden   { background: #1e293b; color: #475569; }
.box-reason { font-size: 10px; color: #64748b; margin-top: 3px; line-height: 1.35; }
.box-coords { font-size: 9px; color: #334155; margin-top: 2px; font-family: monospace; }
.box-actions { display: flex; gap: 3px; margin-top: 5px; flex-wrap: wrap; }
.box-card.is-hidden { opacity: 0.5; }

/* ── Buttons ───────────────────────────────────────────────────────────── */
.btn { font-size: 10px; padding: 2px 8px; border-radius: 3px; border: none;
  cursor: pointer; font-weight: 600; transition: opacity 0.12s; }
.btn:hover { opacity: 0.82; }
.btn-approve { background: #14532d; color: #86efac; }
.btn-reject  { background: #7f1d1d; color: #fca5a5; }
.btn-hide    { background: #27272a; color: #a1a1aa; }
.btn-save   { background: #1d4ed8; color: #fff; padding: 4px 14px; font-size: 12px; border-radius: 4px; }
.btn-nav    { background: #1e293b; color: #e2e8f0; padding: 4px 10px; font-size: 11px;
  border-radius: 4px; border: 1px solid #334155; }
.btn-nav.active { background: #78350f; border-color: #f59e0b; color: #fde68a; }
.badge { font-size: 10px; padding: 2px 8px; border-radius: 10px; background: #1e293b;
  border: 1px solid #334155; color: #94a3b8; }
.save-ok { font-size: 11px; color: #22c55e; font-weight: 600; }
.toolbar h1 { font-size: 14px; font-weight: 700; color: #f8fafc; margin-right: 4px; }
.spacer { flex: 1; }
</style>
</head>
<body>
<div class="app">
  <!-- ── Toolbar ─────────────────────────────────────────────────────────── -->
  <div class="toolbar">
    <h1>Annotation Review</h1>
    <div class="nav-group" id="gridNav">
      <button class="btn btn-nav" id="btn-prev">&#8592; Prev batch <kbd>A</kbd></button>
      <span class="badge" id="batch-counter">0 / 0</span>
      <button class="btn btn-nav" id="btn-next">Next batch <kbd>D</kbd> &#8594;</button>
      <span class="badge" id="total-imgs-badge">0 images</span>
    </div>
    <div class="nav-group" id="detailNav" style="display:none">
      <button class="btn btn-nav" id="btn-back">&#8592; Back to grid <kbd>Esc</kbd></button>
      <button class="btn btn-nav" id="btn-prev-img">&#8592; Prev <kbd>A</kbd></button>
      <span class="badge" id="img-counter">0 / 0</span>
      <button class="btn btn-nav" id="btn-next-img">Next <kbd>D</kbd> &#8594;</button>
      <span class="badge" id="img-id-badge" style="font-family:monospace;font-size:9px;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span>
    </div>
    <div class="spacer"></div>
    <button class="btn btn-nav" id="btn-toggle-all" title="Space">&#128065; Show all invalid</button>
    <span class="save-ok" id="save-ok" style="display:none">&#10003; Saved</span>
    <button class="btn btn-save" id="btn-save">&#128190; Save</button>
  </div>

  <div class="main-row">
    <!-- ── Grid view ─────────────────────────────────────────────────────── -->
    <div class="view" id="gridView">
      <div class="grid-wrap" id="gridWrap">
        <div class="grid" id="grid"></div>
      </div>
    </div>

    <!-- ── Detail view ───────────────────────────────────────────────────── -->
    <div class="view" id="detailView" style="display:none">
      <div class="canvas-panel">
        <div class="canvas-wrapper">
          <canvas id="canvas" tabindex="0"></canvas>
        </div>
        <div class="info-bar">
          <span><b>A/D</b> prev/next image &nbsp; <b>Enter</b> approve &nbsp; <b>X</b> reject &nbsp;
            <b>Space</b> show/hide all &nbsp; <b>drag</b> add box &nbsp; <b>Esc</b> back to grid</span>
          <span id="draw-hint" style="display:none">&#9998; Draw mode &mdash; select class then drag on image</span>
          <span class="spacer"></span>
          <span id="img-url-hint" style="font-size:9px;color:#334155;max-width:300px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span>
        </div>
      </div>

      <div class="box-panel">
        <div class="box-panel-hdr">
          Boxes <span class="badge" id="box-count">0</span>
        </div>
        <div class="cls-section">
          <label>Class for new boxes (then drag on image):</label>
          <select id="cls-sel">
            <option value="">&#8212; select class &#8212;</option>
          </select>
        </div>
        <div class="box-scroll" id="box-list"></div>
      </div>
    </div>
  </div>
</div>

<script>
// ────────────────────────────────────────────────────────────────────────────
//  State
// ────────────────────────────────────────────────────────────────────────────
let images = [];       // [{im_id, im_url, boxes:[{name,box,source,llm_validated,is_valid,reason},...]}]
let classes = [];
let humanReview = { decisions: {}, added_boxes: {}, hidden: {} };

let viewMode = 'grid';  // 'grid' | 'detail'
let curIdx = 0;         // index into full `images` array (detail view)
let selBoxIdx = -1;
let newClass = '';

// Session-only toggle (not persisted): reveals boxes that are hidden by the
// default "invalid boxes are hidden" rule, across every image/batch.
let globalShowAll = false;

// Grid/batch pagination
let gridPage = 0;
let gridCols = 4, gridRows = 3, gridPageSize = 12;

let imgEl = new Image();
let imgOk = false;
let imgScale = 1, imgOX = 0, imgOY = 0;

// drag state
let dragging = false;
let dragP0 = null, dragP1 = null;  // canvas pixel coords

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// ────────────────────────────────────────────────────────────────────────────
//  Boot
// ────────────────────────────────────────────────────────────────────────────
async function boot() {
  const resp = await fetch('/api/data');
  const data = await resp.json();
  images = data.images || [];
  classes = data.classes || [];
  humanReview = data.human_review || { decisions: {}, added_boxes: {}, hidden: {} };
  if (!humanReview.decisions) humanReview.decisions = {};
  if (!humanReview.added_boxes) humanReview.added_boxes = {};
  if (!humanReview.hidden) humanReview.hidden = {};

  // Migrate legacy 'deleted' decisions (from an older version of this tool) into
  // the new hidden-toggle model.
  for (const [uid, dec] of Object.entries(humanReview.decisions)) {
    if (dec === 'deleted') {
      humanReview.hidden[uid] = true;
      delete humanReview.decisions[uid];
    }
  }

  // Populate class selector
  const sel = document.getElementById('cls-sel');
  classes.forEach(c => {
    const o = document.createElement('option');
    o.value = c; o.textContent = c;
    sel.appendChild(o);
  });

  showGrid();
}

// ────────────────────────────────────────────────────────────────────────────
//  Box helpers
// ────────────────────────────────────────────────────────────────────────────
function allBoxes(idx) {
  if (idx < 0 || idx >= images.length) return [];
  const base = (images[idx].boxes || []).map((b, i) => ({ ...b, _src: 'base', _i: i }));
  const added = ((humanReview.added_boxes || {})[images[idx].im_id] || [])
    .map((b, i) => ({ ...b, _src: 'added', _i: i, llm_validated: false, is_valid: true, reason: '' }));
  return [...base, ...added];
}

function boxUid(imId, box) {
  const [x1, y1, x2, y2] = (box.box || [0,0,0,0]).map(v => Math.round(v));
  return `${imId}:${box.name}:${x1}:${y1}:${x2}:${y2}`;
}

// A box is hidden if: (1) it has an explicit per-box override (set by clicking
// its Hide/Show button), otherwise (2) the global "show all" toggle (Space) is
// on, meaning nothing is hidden, otherwise (3) the default rule applies: LLM-
// invalid boxes are hidden by default, everything else is shown.
function isHidden(box, imId) {
  const uid = boxUid(imId, box);
  const overrides = humanReview.hidden || {};
  if (Object.prototype.hasOwnProperty.call(overrides, uid)) return !!overrides[uid];
  if (globalShowAll) return false;
  return box.is_valid === false;
}

// Older pipeline runs tagged the LLM's rejection reason with a bracketed marker
// like "[FLAGGED BLURRY] ...". We no longer treat this as its own category, but
// still strip/show it as a short annotation on the invalid badge for context.
function extractFlagTag(reason) {
  const m = /^\[FLAGGED\s+([^\]]+)\]/i.exec(reason || '');
  return m ? m[1].trim() : null;
}

function stripFlagTag(reason) {
  return (reason || '').replace(/^\[FLAGGED\s+[^\]]+\]\s*/i, '');
}

function effectiveStatus(box, imId) {
  const uid = boxUid(imId, box);
  const dec = (humanReview.decisions || {})[uid];
  if (dec === 'approved') return 'approved';
  if (dec === 'rejected') return 'rejected';
  if (box._src === 'added') return 'new';
  if (box.llm_validated && !box.is_valid) return 'invalid';
  return 'valid';
}

const STATUS_COLOR = {
  valid:    '#22c55e',
  invalid:  '#ef4444',
  approved: '#4ade80',
  rejected: '#f87171',
  new:      '#a78bfa',
};
const STATUS_BADGE = {
  valid:    ['&#10003; valid',    'b-valid'],
  invalid:  ['&#10007; invalid',  'b-invalid'],
  approved: ['&#10003; approved', 'b-approved'],
  rejected: ['&#10007; rejected', 'b-rejected'],
  new:      ['+ new',             'b-new'],
};

function boxColor(status, selected) {
  return selected ? '#38bdf8' : (STATUS_COLOR[status] || '#94a3b8');
}

// ────────────────────────────────────────────────────────────────────────────
//  Grid (batch) view
// ────────────────────────────────────────────────────────────────────────────
const thumbImgCache = {};
function getThumbImg(im_id) {
  if (!thumbImgCache[im_id]) {
    const im = new Image();
    im.src = `/api/image/${encodeURIComponent(im_id)}`;
    thumbImgCache[im_id] = im;
  }
  return thumbImgCache[im_id];
}

function computeGridLayout() {
  const wrap = document.getElementById('gridWrap');
  const w = wrap.clientWidth || 1200;
  const h = wrap.clientHeight || 800;
  const gap = 10;
  const minTileW = 200, minTileH = 190;  // rough target size incl. label bar
  gridCols = Math.max(1, Math.floor((w + gap) / (minTileW + gap)));
  gridRows = Math.max(1, Math.floor((h + gap) / (minTileH + gap)));
  gridPageSize = gridCols * gridRows;
}

function totalGridPages() {
  return Math.max(1, Math.ceil(images.length / gridPageSize));
}

function renderGridPage() {
  computeGridLayout();
  const totalPages = totalGridPages();
  if (gridPage >= totalPages) gridPage = totalPages - 1;
  if (gridPage < 0) gridPage = 0;
  const start = gridPage * gridPageSize;
  const end = Math.min(start + gridPageSize, images.length);

  const gridEl = document.getElementById('grid');
  gridEl.style.gridTemplateColumns = `repeat(${gridCols}, 1fr)`;
  gridEl.style.gridTemplateRows = `repeat(${gridRows}, 1fr)`;
  gridEl.innerHTML = '';

  for (let idx = start; idx < end; idx++) {
    const img = images[idx];
    const tile = document.createElement('div');
    tile.className = 'tile';

    const cnv = document.createElement('canvas');
    tile.appendChild(cnv);

    const lbl = document.createElement('div');
    lbl.className = 'tile-label';
    const invalidHiddenCount = allBoxes(idx).filter(b => b.is_valid === false && isHidden(b, img.im_id)).length;
    lbl.innerHTML = `<span class="t-id">${idx + 1}. ${img.im_id}</span>` +
      (invalidHiddenCount ? `<span class="tile-badge">${invalidHiddenCount} hidden</span>` : '');
    tile.appendChild(lbl);

    tile.onclick = () => openDetail(idx);
    gridEl.appendChild(tile);
    requestAnimationFrame(() => drawTile(cnv, idx));
  }

  document.getElementById('batch-counter').textContent = `${gridPage + 1} / ${totalPages}`;
  document.getElementById('total-imgs-badge').textContent = `${images.length} images`;
}

function drawTile(cnv, idx) {
  const parent = cnv.parentElement;
  if (!parent) return;
  const rect = parent.getBoundingClientRect();
  const labelH = 20;
  const w = Math.max(10, Math.round(rect.width));
  const h = Math.max(10, Math.round(rect.height - labelH));
  cnv.width = w;
  cnv.height = h;
  const c = cnv.getContext('2d');
  const img = images[idx];
  const imEl = getThumbImg(img.im_id);

  function draw() {
    c.clearRect(0, 0, w, h);
    c.fillStyle = '#0a0f1e';
    c.fillRect(0, 0, w, h);
    if (!imEl.naturalWidth) return;
    const scale = Math.min(w / imEl.naturalWidth, h / imEl.naturalHeight);
    const ox = (w - imEl.naturalWidth * scale) / 2;
    const oy = (h - imEl.naturalHeight * scale) / 2;
    c.drawImage(imEl, ox, oy, imEl.naturalWidth * scale, imEl.naturalHeight * scale);
    allBoxes(idx).forEach(b => {
      if (isHidden(b, img.im_id)) return;
      const st = effectiveStatus(b, img.im_id);
      const col = STATUS_COLOR[st] || '#94a3b8';
      const [x1, y1, x2, y2] = b.box || [0, 0, 0, 0];
      c.strokeStyle = col;
      c.lineWidth = 1.5;
      c.strokeRect(ox + x1 * scale, oy + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
    });
  }
  if (imEl.complete && imEl.naturalWidth) draw();
  else imEl.onload = draw;
}

function showGrid() {
  viewMode = 'grid';
  document.getElementById('gridView').style.display = '';
  document.getElementById('detailView').style.display = 'none';
  document.getElementById('gridNav').style.display = '';
  document.getElementById('detailNav').style.display = 'none';
  renderGridPage();
}

function openDetail(idx) {
  viewMode = 'detail';
  document.getElementById('gridView').style.display = 'none';
  document.getElementById('detailView').style.display = '';
  document.getElementById('gridNav').style.display = 'none';
  document.getElementById('detailNav').style.display = '';
  requestAnimationFrame(() => {
    resizeCanvas();
    loadImg(idx);
  });
}

function closeDetail() {
  gridPage = Math.floor(curIdx / gridPageSize);
  showGrid();
}

// ────────────────────────────────────────────────────────────────────────────
//  Load image (detail view)
// ────────────────────────────────────────────────────────────────────────────
function loadImg(idx) {
  curIdx = idx;
  selBoxIdx = -1;
  imgOk = false;
  render();

  const img = images[idx];
  document.getElementById('img-counter').textContent = `${idx + 1} / ${images.length}`;
  document.getElementById('img-id-badge').textContent = img.im_id;
  document.getElementById('img-url-hint').textContent = img.im_url || '';

  imgEl = new Image();
  imgEl.onload = () => { imgOk = true; computeTransform(); render(); buildBoxList(); };
  imgEl.onerror = () => {
    imgOk = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = '#0a0f1e'; ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle = '#ef4444'; ctx.font='13px sans-serif'; ctx.textAlign='center';
    ctx.fillText('Failed to load image: ' + img.im_id, canvas.width/2, canvas.height/2);
    ctx.textAlign='left';
    buildBoxList();
  };
  imgEl.src = `/api/image/${encodeURIComponent(img.im_id)}`;

  buildBoxList();
}

// ────────────────────────────────────────────────────────────────────────────
//  Canvas render (detail view)
// ────────────────────────────────────────────────────────────────────────────
function resizeCanvas() {
  // canvas is absolutely positioned inside .canvas-wrapper, read wrapper dims
  const wrapper = canvas.parentElement;
  const w = wrapper.clientWidth || 800;
  const h = wrapper.clientHeight || 600;
  if (w === 0 || h === 0) return;
  canvas.width = w;
  canvas.height = h;
  computeTransform();
  render();
}

function computeTransform() {
  if (!imgOk) return;
  const iw = imgEl.naturalWidth, ih = imgEl.naturalHeight;
  const cw = canvas.width, ch = canvas.height;
  imgScale = Math.min(cw / iw, ch / ih) * 0.97;
  imgOX = (cw - iw * imgScale) / 2;
  imgOY = (ch - ih * imgScale) / 2;
}

function render() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Background
  ctx.fillStyle = '#0a0f1e';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (!imgOk) {
    ctx.fillStyle = '#334155';
    ctx.font = '14px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Loading image…', canvas.width / 2, canvas.height / 2);
    ctx.textAlign = 'left';
    return;
  }

  ctx.drawImage(imgEl, imgOX, imgOY, imgEl.naturalWidth * imgScale, imgEl.naturalHeight * imgScale);

  const imId = images[curIdx]?.im_id || '';
  const boxes = allBoxes(curIdx);

  boxes.forEach((box, i) => {
    const st = effectiveStatus(box, imId);
    if (isHidden(box, imId)) return;

    const [x1, y1, x2, y2] = (box.box || [0,0,0,0]);
    const cx1 = imgOX + x1 * imgScale;
    const cy1 = imgOY + y1 * imgScale;
    const cw  = (x2 - x1) * imgScale;
    const ch  = (y2 - y1) * imgScale;

    const sel = i === selBoxIdx;
    const col = boxColor(st, sel);

    // Box outline (selected gets a thicker white glow)
    if (sel) {
      ctx.strokeStyle = 'rgba(255,255,255,0.4)';
      ctx.lineWidth = 6;
      ctx.strokeRect(cx1, cy1, cw, ch);
    }
    ctx.strokeStyle = col;
    ctx.lineWidth = sel ? 2.5 : 1.5;
    ctx.strokeRect(cx1, cy1, cw, ch);

    // Label
    ctx.font = 'bold 11px sans-serif';
    const lbl = box.name || '?';
    const tw = ctx.measureText(lbl).width + 6;
    const labelAbove = cy1 > 16;
    const labelTop = labelAbove ? cy1 - 15 : cy1 + 1;

    ctx.fillStyle = col;
    ctx.globalAlpha = 0.85;
    ctx.fillRect(cx1, labelTop, tw, 14);
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#fff';
    ctx.fillText(lbl, cx1 + 3, labelTop + 11);
  });

  // Drag preview
  if (dragging && dragP0 && dragP1) {
    const x = Math.min(dragP0.x, dragP1.x);
    const y = Math.min(dragP0.y, dragP1.y);
    const w = Math.abs(dragP1.x - dragP0.x);
    const h = Math.abs(dragP1.y - dragP0.y);
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 3]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    // Corner label
    ctx.font = 'bold 11px sans-serif';
    const lbl = newClass || '?';
    ctx.fillStyle = 'rgba(245,158,11,0.85)';
    ctx.fillRect(x, y - 15, ctx.measureText(lbl).width + 6, 14);
    ctx.fillStyle = '#fff';
    ctx.fillText(lbl, x + 3, y - 4);
  }
}

// ────────────────────────────────────────────────────────────────────────────
//  Box list (right panel, detail view)
// ────────────────────────────────────────────────────────────────────────────
function buildBoxList() {
  const el = document.getElementById('box-list');
  const imId = images[curIdx]?.im_id || '';
  const boxes = allBoxes(curIdx);
  const visible = boxes.filter(b => !isHidden(b, imId));
  document.getElementById('box-count').textContent = visible.length;

  el.innerHTML = '';
  boxes.forEach((box, i) => {
    const st = effectiveStatus(box, imId);
    const hidden = isHidden(box, imId);
    let [bl, bc] = STATUS_BADGE[st] || ['?', ''];
    const col = boxColor(st, false);
    const sel = i === selBoxIdx;
    const [x1, y1, x2, y2] = (box.box || [0,0,0,0]).map(v => Math.round(v));

    const flagTag = extractFlagTag(box.reason);
    if (flagTag) {
      bl = `${bl} &middot; ${flagTag.toLowerCase()}`;
    }
    const reasonText = stripFlagTag(box.reason);

    const div = document.createElement('div');
    div.className = 'box-card' + (sel ? ' selected' : '') + (hidden ? ' is-hidden' : '');
    div.innerHTML = `
      <div class="box-card-top">
        <span class="box-name" style="color:${col}">${box.name || '?'}</span>
        <span class="box-badge ${hidden ? 'b-hidden' : bc}">${hidden ? '&#128065; hidden' : bl}</span>
      </div>
      ${reasonText ? `<div class="box-reason">${reasonText}</div>` : ''}
      <div class="box-coords">[${x1}, ${y1}, ${x2}, ${y2}]</div>
      <div class="box-actions">
        <button class="btn btn-approve" title="Approve (Enter)" data-action="approve" data-i="${i}">&#10003;</button>
        <button class="btn btn-reject"  title="Reject (X)"       data-action="reject"  data-i="${i}">&#10007;</button>
        <button class="btn btn-hide"    title="Hide/Show (click)" data-action="toggle-hide" data-i="${i}">${hidden ? '&#128065; Show' : '&#128683; Hide'}</button>
      </div>`;
    div.addEventListener('click', e => {
      const btn = e.target.closest('[data-action]');
      if (btn) {
        e.stopPropagation();
        const action = btn.dataset.action;
        const bi = parseInt(btn.dataset.i, 10);
        if (action === 'approve') doApprove(bi);
        else if (action === 'reject') doReject(bi);
        else if (action === 'toggle-hide') doToggleHide(bi);
        return;
      }
      doSelect(i);
    });
    el.appendChild(div);
    if (sel) div.scrollIntoView({ block: 'nearest' });
  });
}

// ────────────────────────────────────────────────────────────────────────────
//  Actions
// ────────────────────────────────────────────────────────────────────────────
function doSelect(i) {
  selBoxIdx = i;
  render();
  buildBoxList();
}

function doApprove(i) {
  const imId = images[curIdx]?.im_id || '';
  const box = allBoxes(curIdx)[i];
  if (!box) return;
  const uid = boxUid(imId, box);
  humanReview.decisions[uid] = humanReview.decisions[uid] === 'approved' ? undefined : 'approved';
  if (humanReview.decisions[uid] === undefined) delete humanReview.decisions[uid];
  render();
  buildBoxList();
}

function doReject(i) {
  const imId = images[curIdx]?.im_id || '';
  const box = allBoxes(curIdx)[i];
  if (!box) return;
  const uid = boxUid(imId, box);
  humanReview.decisions[uid] = humanReview.decisions[uid] === 'rejected' ? undefined : 'rejected';
  if (humanReview.decisions[uid] === undefined) delete humanReview.decisions[uid];
  render();
  buildBoxList();
}

// Toggles this specific box's visibility, pinning an explicit override that
// takes precedence over the default-hide-invalid rule and the global toggle.
function doToggleHide(i) {
  const imId = images[curIdx]?.im_id || '';
  const box = allBoxes(curIdx)[i];
  if (!box) return;
  const uid = boxUid(imId, box);
  humanReview.hidden[uid] = !isHidden(box, imId);
  render();
  buildBoxList();
}

// Space bar: bulk-toggle visibility of every box (in the current batch and in
// the single-image detail view) that isn't individually overridden.
function doToggleShowAll() {
  globalShowAll = !globalShowAll;
  const btn = document.getElementById('btn-toggle-all');
  btn.innerHTML = globalShowAll ? '&#128584; Hide invalid' : '&#128065; Show all invalid';
  btn.classList.toggle('active', globalShowAll);
  if (viewMode === 'grid') {
    renderGridPage();
  } else {
    render();
    buildBoxList();
  }
}

// ────────────────────────────────────────────────────────────────────────────
//  Canvas mouse: click-to-select OR click-drag to draw (detail view)
// ────────────────────────────────────────────────────────────────────────────
canvas.addEventListener('mousedown', e => {
  const r = canvas.getBoundingClientRect();
  const mx = e.clientX - r.left, my = e.clientY - r.top;

  if (newClass) {
    // Draw mode
    dragging = true;
    dragP0 = { x: mx, y: my };
    dragP1 = { x: mx, y: my };
    return;
  }

  // Click-to-select
  if (!imgOk) return;
  const ix = (mx - imgOX) / imgScale;
  const iy = (my - imgOY) / imgScale;
  const imId = images[curIdx]?.im_id || '';
  const boxes = allBoxes(curIdx);
  let found = -1;
  for (let i = boxes.length - 1; i >= 0; i--) {
    if (isHidden(boxes[i], imId)) continue;
    const [x1, y1, x2, y2] = boxes[i].box || [0,0,0,0];
    if (ix >= x1 && ix <= x2 && iy >= y1 && iy <= y2) { found = i; break; }
  }
  selBoxIdx = found;
  render();
  buildBoxList();
});

canvas.addEventListener('mousemove', e => {
  if (!dragging) return;
  const r = canvas.getBoundingClientRect();
  dragP1 = { x: e.clientX - r.left, y: e.clientY - r.top };
  render();
});

canvas.addEventListener('mouseup', e => {
  if (!dragging) return;
  dragging = false;

  const dx = Math.abs((dragP1?.x || 0) - (dragP0?.x || 0));
  const dy = Math.abs((dragP1?.y || 0) - (dragP0?.y || 0));
  if (dx < 5 && dy < 5) { dragP0 = dragP1 = null; render(); return; }

  if (!imgOk) { dragP0 = dragP1 = null; return; }
  const iw = imgEl.naturalWidth, ih = imgEl.naturalHeight;
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
  function toImg(cx, cy) {
    return [(cx - imgOX) / imgScale, (cy - imgOY) / imgScale];
  }
  const [ax, ay] = toImg(Math.min(dragP0.x, dragP1.x), Math.min(dragP0.y, dragP1.y));
  const [bx, by] = toImg(Math.max(dragP0.x, dragP1.x), Math.max(dragP0.y, dragP1.y));
  const newBox = {
    box: [
      Math.round(clamp(ax, 0, iw)), Math.round(clamp(ay, 0, ih)),
      Math.round(clamp(bx, 0, iw)), Math.round(clamp(by, 0, ih)),
    ],
    name: newClass,
    source: 'human',
    llm_validated: false, is_valid: true, reason: '',
    _src: 'added', _i: 0,
  };
  const imId = images[curIdx].im_id;
  if (!humanReview.added_boxes[imId]) humanReview.added_boxes[imId] = [];
  humanReview.added_boxes[imId].push(newBox);

  dragP0 = dragP1 = null;
  selBoxIdx = allBoxes(curIdx).length - 1;  // select the new box
  render();
  buildBoxList();
});

// ────────────────────────────────────────────────────────────────────────────
//  Keyboard hotkeys
// ────────────────────────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (document.activeElement.tagName === 'SELECT' ||
      document.activeElement.tagName === 'INPUT') return;

  if (e.key === ' ') {
    e.preventDefault();
    doToggleShowAll();
    return;
  }

  if (viewMode === 'grid') {
    switch (e.key) {
      case 'a': case 'A': case 'ArrowLeft':
        e.preventDefault();
        if (gridPage > 0) { gridPage--; renderGridPage(); }
        break;
      case 'd': case 'D': case 'ArrowRight':
        e.preventDefault();
        if (gridPage < totalGridPages() - 1) { gridPage++; renderGridPage(); }
        break;
    }
    return;
  }

  // Detail view
  switch (e.key) {
    case 'a': case 'A': case 'ArrowLeft':
      e.preventDefault();
      if (curIdx > 0) loadImg(curIdx - 1);
      break;
    case 'd': case 'D': case 'ArrowRight':
      e.preventDefault();
      if (curIdx < images.length - 1) loadImg(curIdx + 1);
      break;
    case 'Enter':
      e.preventDefault();
      if (selBoxIdx >= 0) doApprove(selBoxIdx);
      break;
    case 'x': case 'X':
      if (selBoxIdx >= 0) { e.preventDefault(); doReject(selBoxIdx); }
      break;
    case 'Escape':
      e.preventDefault();
      if (newClass) {
        newClass = '';
        document.getElementById('cls-sel').value = '';
        document.getElementById('draw-hint').style.display = 'none';
        canvas.style.cursor = 'default';
        dragging = false; dragP0 = dragP1 = null;
        render();
      } else {
        closeDetail();
      }
      break;
  }
});

// ────────────────────────────────────────────────────────────────────────────
//  Class selector
// ────────────────────────────────────────────────────────────────────────────
document.getElementById('cls-sel').addEventListener('change', function () {
  newClass = this.value;
  document.getElementById('draw-hint').style.display = newClass ? '' : 'none';
  canvas.style.cursor = newClass ? 'crosshair' : 'default';
});

// ────────────────────────────────────────────────────────────────────────────
//  Save
// ────────────────────────────────────────────────────────────────────────────
document.getElementById('btn-save').addEventListener('click', saveReview);

async function saveReview() {
  // Strip runtime-only fields (_src, _i) from added_boxes before serialising
  const toSave = {
    decisions: humanReview.decisions,
    added_boxes: {},
    hidden: humanReview.hidden,
  };
  for (const [imId, arr] of Object.entries(humanReview.added_boxes)) {
    toSave.added_boxes[imId] = arr.map(b => ({
      box: b.box, name: b.name, source: 'human',
    }));
  }
  const resp = await fetch('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(toSave),
  });
  if (resp.ok) {
    const ok = document.getElementById('save-ok');
    ok.style.display = '';
    setTimeout(() => { ok.style.display = 'none'; }, 2500);
  }
}

// ────────────────────────────────────────────────────────────────────────────
//  Navigation buttons
// ────────────────────────────────────────────────────────────────────────────
document.getElementById('btn-prev').addEventListener('click', () => {
  if (gridPage > 0) { gridPage--; renderGridPage(); }
});
document.getElementById('btn-next').addEventListener('click', () => {
  if (gridPage < totalGridPages() - 1) { gridPage++; renderGridPage(); }
});
document.getElementById('btn-back').addEventListener('click', closeDetail);
document.getElementById('btn-prev-img').addEventListener('click', () => {
  if (curIdx > 0) loadImg(curIdx - 1);
});
document.getElementById('btn-next-img').addEventListener('click', () => {
  if (curIdx < images.length - 1) loadImg(curIdx + 1);
});
document.getElementById('btn-toggle-all').addEventListener('click', doToggleShowAll);

// ────────────────────────────────────────────────────────────────────────────
//  Resize
// ────────────────────────────────────────────────────────────────────────────
const gridRO = new ResizeObserver(() => { if (viewMode === 'grid') renderGridPage(); });
gridRO.observe(document.getElementById('gridWrap'));

const canvasRO = new ResizeObserver(() => { if (viewMode === 'detail') resizeCanvas(); });
canvasRO.observe(canvas.parentElement);  // observes .canvas-wrapper

// ────────────────────────────────────────────────────────────────────────────
//  Go
// ────────────────────────────────────────────────────────────────────────────
boot();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP Server
# ---------------------------------------------------------------------------

class _ReviewHandler(BaseHTTPRequestHandler):
    # Class-level shared state set by main() before starting
    images_data: list = []
    classes: list = []
    human_review: dict = {}
    image_cache_dir: Path = Path(".")
    review_path: Path = Path("human_review.json")

    def log_message(self, fmt, *args):  # silence default request logging
        pass

    # ── GET ─────────────────────────────────────────────────────────────────
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        p = parsed.path.rstrip("/") or "/"

        if p in ("/", "/index.html"):
            self._send_bytes(_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif p == "/api/data":
            # Strip server-side-only fields before sending to browser
            browser_images = [
                {k: v for k, v in img.items() if not k.startswith("_")}
                for img in self.images_data
            ]
            payload = json.dumps({
                "images": browser_images,
                "classes": self.classes,
                "human_review": self.human_review,
            }).encode("utf-8")
            self._send_bytes(payload, "application/json")
        elif p.startswith("/api/image/"):
            im_id = urllib.parse.unquote(p[len("/api/image/"):])
            self._serve_image(im_id)
        elif p == "/api/review":
            self._send_bytes(json.dumps(self.human_review).encode(), "application/json")
        else:
            self.send_error(404, "Not found")

    # ── POST ────────────────────────────────────────────────────────────────
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        p = parsed.path

        if p == "/api/save":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                self.review_path.parent.mkdir(parents=True, exist_ok=True)
                self.review_path.write_text(json.dumps(data, indent=2))
                # Sync back to class-level so further GET /api/review is current
                _ReviewHandler.human_review = data
                logger.info("Saved human review → {}", self.review_path)
                self._send_bytes(b'{"ok":true}', "application/json")
            except Exception as exc:
                logger.error("Save failed: {}", exc)
                self.send_error(500, str(exc))
        else:
            self.send_error(404, "Not found")

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _serve_image(self, im_id: str):
        entry = next((x for x in self.images_data if x["im_id"] == im_id), None)

        # Try candidates in priority order: im_id.jpg, then the URL's MD5 hash.jpg
        candidates: list[Path] = [self.image_cache_dir / f"{im_id}.jpg"]
        if entry:
            cache_key = entry.get("_img_cache_key", "")
            if cache_key and cache_key != im_id:
                candidates.append(self.image_cache_dir / f"{cache_key}.jpg")

        for path in candidates:
            if path.exists():
                self._send_bytes(path.read_bytes(), "image/jpeg")
                return

        # Last resort: download and cache
        if entry:
            try:
                _, local_path = load_image_and_path(entry.get("im_url", ""), self.image_cache_dir)
                if local_path and Path(local_path).exists():
                    self._send_bytes(Path(local_path).read_bytes(), "image/jpeg")
                    return
            except Exception as exc:
                logger.warning("Could not fetch image {}: {}", im_id, exc)

        self.send_error(404, f"Image {im_id} not found")

    def _send_bytes(self, data: bytes, content_type: str):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    config_file: str = typer.Argument(..., help="Path to auto-annotate config YAML"),
    port: int = typer.Option(7654, help="Local port for the review server"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Do not auto-open browser"),
):
    """Launch the interactive annotation review GUI in your browser.

    Reads the pipeline output from processed_cache.json + llm_cache.json,
    serves a browser-based review tool, and saves decisions to human_review.json.
    """
    images_data, image_cache_dir, review_path = load_pipeline_data(config_file)

    human_review: dict[str, Any] = {"decisions": {}, "added_boxes": {}, "hidden": {}}
    if review_path.exists():
        try:
            human_review = json.loads(review_path.read_text())
        except Exception as exc:
            logger.warning("Could not load existing human_review.json: {}", exc)

    classes = load_classes()

    _ReviewHandler.images_data = images_data
    _ReviewHandler.classes = classes
    _ReviewHandler.human_review = human_review
    _ReviewHandler.image_cache_dir = image_cache_dir
    _ReviewHandler.review_path = review_path

    url = f"http://localhost:{port}"
    logger.info("Annotation review server → {}", url)
    logger.info("  {} images loaded, {} classes", len(images_data), len(classes))
    logger.info("  Human review file: {}", review_path)
    logger.info("  Press Ctrl+C to stop")

    if not no_browser:
        import threading
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    with _ThreadingHTTPServer(("localhost", port), _ReviewHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("Server stopped.")


@app.command(name="export")
def export_reviewed(
    config_file: str = typer.Argument(..., help="Path to auto-annotate config YAML"),
):
    """Apply saved human review decisions (human_review.json) on top of the
    pipeline cache and write a new '{dataset}_human_reviewed.csv', without
    modifying the original '{dataset}_annotated.csv' produced by `auto_annotate.py main`.

    Per box: explicit human reject -> excluded; explicit human approve ->
    included (overrides the LLM verdict); otherwise included iff the LLM (or
    lack of LLM validation) already marked it valid. Hiding a box in the review
    UI is just a visibility toggle and does not exclude it. Human-added boxes
    are always included.
    """
    build_human_reviewed_csv(config_file)


if __name__ == "__main__":
    app()
