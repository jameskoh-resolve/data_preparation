#!/usr/bin/env python
"""Generate a rich HTML visualization of a curated dataset.

Reads human_worn_sample.csv (accepted images) and optionally a pool CSV to
enrich with brand/colour metadata. Produces an interactive gallery with:
  - Category filter pills
  - Tag filter pills (human_angle, image_detail, image_human)
  - Per-category mini bar chart
  - Card-level product info: title, brand, colour, category, tags

Usage:
    python scripts/curation/generate_sample_visualization.py <output_dir> [--pool-csv <path>]
"""

import argparse
import ast
import json
import math
import os
import sys

import pandas as pd
from path import Path

REPO_ROOT = Path(os.path.abspath(__file__)).parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Tag colour palette ─────────────────────────────────────────────────────────
TAG_COLORS: dict[str, tuple[str, str]] = {
    "image_human:model": ("#10b981", "#065f46"),
    "image_human:no_model": ("#f43f5e", "#881337"),
    "human_angle:front_view": ("#6366f1", "#312e81"),
    "human_angle:front_angled_view": ("#8b5cf6", "#3b0764"),
    "human_angle:back_view": ("#f59e0b", "#78350f"),
    "human_angle:side_view": ("#ec4899", "#831843"),
    "human_angle:zoom_in_view": ("#14b8a6", "#134e4a"),
    "image_detail:no_detail": ("#64748b", "#0f172a"),
    "image_detail:detail": ("#0ea5e9", "#0c4a6e"),
}
DEFAULT_TAG_COLOR = ("#94a3b8", "#1e293b")

# Stable priority order for tag filter display
TAG_PRIORITY = [
    "image_human:model", "image_human:no_model",
    "human_angle:front_view", "human_angle:front_angled_view",
    "human_angle:back_view", "human_angle:side_view", "human_angle:zoom_in_view",
    "image_detail:no_detail", "image_detail:detail",
]

# One accent colour per category (cycles through a palette)
CATEGORY_PALETTE = [
    "#6366f1", "#f59e0b", "#10b981", "#ec4899", "#14b8a6",
    "#f43f5e", "#8b5cf6", "#0ea5e9", "#84cc16", "#fb923c",
    "#a78bfa", "#34d399", "#f472b6", "#38bdf8", "#fbbf24",
]


def _category_str(raw: str) -> str:
    """Normalise a category cell that may be a stringified list."""
    if not raw or raw == "nan":
        return ""
    raw = str(raw).strip()
    if raw.startswith("["):
        try:
            parts = ast.literal_eval(raw)
            if isinstance(parts, list):
                return ", ".join(str(p).strip("' ") for p in parts)
        except Exception:
            pass
    return raw


def _tag_chip(tag: str) -> str:
    bg, _ = TAG_COLORS.get(tag, DEFAULT_TAG_COLOR)
    label = tag.split(":")[-1].replace("_", " ")
    return (
        f'<span class="tag-chip" data-tag="{tag}" style="background:{bg}">'
        f"{label}</span>"
    )


def _safe_str(val) -> str:
    s = str(val) if val is not None else ""
    return "" if s in ("nan", "None") else s.strip()


def _dataset_label_from_output_dir(out_path: Path) -> str:
    """Build a readable dataset label from the output directory path."""
    parent = str(out_path.parent.name or "").replace("_", " ").replace("-", " ").strip()
    leaf = str(out_path.name or "").replace("_", " ").replace("-", " ").strip()
    if leaf.endswith(" sample"):
        leaf = leaf[: -len(" sample")].strip()
    label = " ".join([x for x in [parent, leaf] if x]).strip()
    return label if label else "Curated Dataset"


def generate_html(output_dir: str, pool_csv: str | None = None) -> None:
    out_path = Path(output_dir)
    dataset_label = _dataset_label_from_output_dir(out_path)
    sample_csv = out_path / "human_worn_sample.csv"
    stats_json = out_path / "human_worn_sampling_stats.json"

    if not sample_csv.exists():
        raise FileNotFoundError(
            f"human_worn_sample.csv not found in {output_dir}. Run the sampling pipeline first."
        )

    df = pd.read_csv(sample_csv, dtype={"product_id": str})

    # ── Enrich with pool metadata (brand, colour, title) ──────────────────────
    if pool_csv:
        pool_path = Path(pool_csv) if Path(pool_csv).isabs() else REPO_ROOT / pool_csv
        if pool_path.exists():
            pool_df = pd.read_csv(pool_path, dtype={"product_id": str})
            enrich_cols = [c for c in ["im_url", "brand", "colour", "title"] if c in pool_df.columns]
            if enrich_cols:
                df = df.merge(pool_df[enrich_cols], on="im_url", how="left")

    # ── Normalise columns ─────────────────────────────────────────────────────
    df["tags_list"] = df.get("tags", pd.Series([""] * len(df))).fillna("").apply(
        lambda t: [x.strip() for x in t.split(",") if x.strip()]
    )
    df["category_clean"] = df.get("category", pd.Series([""] * len(df))).fillna("").apply(_category_str)
    df["title_clean"] = df.get("im_name", df.get("title", pd.Series([""] * len(df)))).apply(_safe_str)
    df["brand_clean"] = df.get("brand", pd.Series([""] * len(df))).apply(_safe_str)
    df["colour_clean"] = df.get("colour", pd.Series([""] * len(df))).apply(_safe_str)

    # ── Collect all unique tags ───────────────────────────────────────────────
    all_tags: list[str] = []
    seen_tags: set[str] = set()
    for pt in TAG_PRIORITY:
        if pt not in seen_tags and df["tags_list"].apply(lambda tl: pt in tl).any():
            all_tags.append(pt)
            seen_tags.add(pt)
    for tl in df["tags_list"]:
        for t in tl:
            if t not in seen_tags:
                all_tags.append(t)
                seen_tags.add(t)

    # ── Collect all unique categories ─────────────────────────────────────────
    all_categories = sorted(df["category_clean"].dropna().unique().tolist())
    all_categories = [c for c in all_categories if c]
    cat_color_map = {c: CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)] for i, c in enumerate(sorted(all_categories))}

    # ── Category bar chart data ───────────────────────────────────────────────
    cat_counts = df["category_clean"].value_counts().to_dict()
    max_count = max(cat_counts.values()) if cat_counts else 1
    cat_bars_html = ""
    for cat in sorted(cat_counts.keys()):
        count = cat_counts[cat]
        pct = count / max_count * 100
        color = cat_color_map.get(cat, "#6366f1")
        cat_bars_html += f"""
        <div class="bar-row">
            <span class="bar-label">{cat}</span>
            <div class="bar-track">
                <div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
            </div>
            <span class="bar-count">{count}</span>
        </div>"""

    # ── Stats panel ───────────────────────────────────────────────────────────
    stats_html = f"""
    <div class="stat-card">
        <h3>Total Images</h3>
        <div class="stat-value">{len(df)}</div>
    </div>
    <div class="stat-card accent-green">
        <h3>Categories</h3>
        <div class="stat-value">{len(all_categories)}</div>
    </div>
    <div class="stat-card accent-purple">
        <h3>Unique Tags</h3>
        <div class="stat-value">{len(all_tags)}</div>
    </div>"""
    if stats_json.exists():
        try:
            stats_data = json.loads(stats_json.read_text(encoding="utf-8"))
            total_eval = sum(c.get("evaluated", 0) for c in stats_data.values())
            total_acc = sum(c.get("accepted", 0) for c in stats_data.values())
            acc_rate = (total_acc / total_eval * 100) if total_eval > 0 else 0.0
            stats_html += f"""
    <div class="stat-card">
        <h3>Acceptance Rate</h3>
        <div class="stat-value">{acc_rate:.1f}%</div>
        <div class="stat-sub">{total_eval} evaluated</div>
    </div>"""
        except Exception:
            pass

    # ── Build image cards ─────────────────────────────────────────────────────
    cards_html_parts: list[str] = []
    for _, row in df.iterrows():
        tags_list = row["tags_list"]
        im_url = _safe_str(row.get("im_url", ""))
        product_id = _safe_str(row.get("product_id", ""))
        category = row["category_clean"]
        title = row["title_clean"]
        brand = row["brand_clean"]
        colour = row["colour_clean"]

        cat_color = cat_color_map.get(category, "#6366f1")
        chips_html = " ".join(_tag_chip(t) for t in tags_list)
        tags_json = json.dumps(tags_list)
        cat_attr = f'data-category="{category}"' if category else ""

        brand_badge = f'<span class="meta-pill brand-pill">{brand}</span>' if brand else ""
        colour_badge = f'<span class="meta-pill colour-pill" style="--swatch:{colour.lower()}">{colour}</span>' if colour else ""

        title_display = title if title else "—"
        fallback_title = title.replace('"', '&quot;') if title else ""

        cards_html_parts.append(f"""
        <div class="card" {cat_attr} data-tags='{tags_json}' data-category-val="{category}">
            <div class="card-image-container">
                <img class="card-image" src="{im_url}" alt="{fallback_title}" loading="lazy"
                     onerror="this.src='https://placehold.co/400x500/1e293b/94a3b8?text=Load+Error'">
                <div class="card-overlay">
                    <span class="cat-badge" style="background:{cat_color}22;border-color:{cat_color}44;color:{cat_color}">{category}</span>
                </div>
            </div>
            <div class="card-content">
                <div class="product-meta-row">
                    <span class="product-id">#{product_id}</span>
                    {brand_badge}
                    {colour_badge}
                </div>
                <h4 class="product-title" title="{fallback_title}">{title_display}</h4>
                <div class="chips">{chips_html}</div>
                <div class="card-footer">
                    <a href="{im_url}" target="_blank" rel="noopener" class="btn-link">
                        View image
                        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6M15 3h6v6M10 14L21 3"/>
                        </svg>
                    </a>
                </div>
            </div>
        </div>""")

    all_cards_html = "\n".join(cards_html_parts)

    # ── Tag filter buttons ────────────────────────────────────────────────────
    tag_filter_btns = ""
    for tag in all_tags:
        bg, _ = TAG_COLORS.get(tag, DEFAULT_TAG_COLOR)
        label = tag.split(":")[-1].replace("_", " ")
        tag_filter_btns += (
            f'<button class="filter-btn tag-filter-btn" data-tag="{tag}" '
            f'style="--tag-color:{bg}" title="{tag}">{label}</button>\n'
        )

    # ── Category filter buttons ───────────────────────────────────────────────
    cat_filter_btns = ""
    for cat in sorted(all_categories):
        color = cat_color_map.get(cat, "#6366f1")
        count = cat_counts.get(cat, 0)
        cat_filter_btns += (
            f'<button class="filter-btn cat-filter-btn" data-category="{cat}" '
            f'style="--tag-color:{color}">{cat} <span class="btn-count">{count}</span></button>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{dataset_label} Dataset – Curated Gallery</title>
<meta name="description" content="Interactive gallery of {len(df)} curated images with category and tag filtering.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {{
    --bg: #080f1e;
    --surface: #101828;
    --surface2: #162033;
    --card-bg: #1a2540;
    --border: #1f3050;
    --border2: #2a3f63;
    --text: #f1f5f9;
    --muted: #64748b;
    --muted2: #94a3b8;
    --primary: #6366f1;
    --green: #10b981;
    --red: #f43f5e;
    --purple: #8b5cf6;
    --shadow-sm: 0 4px 16px -4px rgba(0,0,0,.5);
    --shadow: 0 8px 32px -8px rgba(0,0,0,.6);
    --shadow-lg: 0 20px 48px -12px rgba(0,0,0,.75);
    --ease: cubic-bezier(.4,0,.2,1);
    --radius: 1rem;
}}
*, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:'Outfit',sans-serif; background:var(--bg); color:var(--text); min-height:100vh; }}

/* ── Layout ─────────────────────────────────────────────────────────── */
.layout {{ display:grid; grid-template-columns:300px 1fr; min-height:100vh; }}
.sidebar {{
    background:var(--surface); border-right:1px solid var(--border);
    padding:1.5rem 1.25rem; position:sticky; top:0; height:100vh;
    overflow-y:auto; display:flex; flex-direction:column; gap:1.5rem;
    scrollbar-width:thin; scrollbar-color:var(--border2) transparent;
}}
.main {{ padding:1.75rem; overflow:hidden; }}

/* ── Logo / header ────────────────────────────────────────────────────── */
.logo {{ padding-bottom:1rem; border-bottom:1px solid var(--border); }}
.logo h1 {{
    font-size:1.35rem; font-weight:700; letter-spacing:-.02em;
    background:linear-gradient(135deg,#a5b4fc,#6366f1 60%,#8b5cf6);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.logo p {{ color:var(--muted); font-size:.8rem; margin-top:.25rem; }}

/* ── Stats ────────────────────────────────────────────────────────────── */
.stats-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:.75rem; }}
.stat-card {{
    background:var(--surface2); border:1px solid var(--border);
    border-radius:.75rem; padding:1rem; text-align:center; position:relative; overflow:hidden;
}}
.stat-card::before {{ content:''; position:absolute; top:0; left:0; right:0; height:2px; background:var(--primary); }}
.stat-card.accent-green::before {{ background:var(--green); }}
.stat-card.accent-purple::before {{ background:var(--purple); }}
.stat-card h3 {{ font-size:.65rem; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); margin-bottom:.3rem; }}
.stat-value {{ font-size:1.6rem; font-weight:700; }}
.stat-sub {{ font-size:.7rem; color:var(--muted); margin-top:.1rem; }}

/* ── Filter sections ──────────────────────────────────────────────────── */
.filter-section-title {{
    font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.1em;
    color:var(--muted); margin-bottom:.6rem; display:flex; align-items:center; justify-content:space-between;
}}
.filter-pills {{ display:flex; flex-wrap:wrap; gap:.4rem; }}
.filter-btn {{
    background:transparent;
    border:1.5px solid var(--tag-color, var(--border2));
    color:var(--tag-color, var(--muted2));
    font-family:inherit; font-size:.75rem; font-weight:500;
    padding:.3rem .75rem; border-radius:9999px; cursor:pointer;
    transition:all .18s var(--ease); white-space:nowrap;
    display:inline-flex; align-items:center; gap:.3rem;
}}
.filter-btn:hover {{ background:color-mix(in srgb, var(--tag-color) 12%, transparent); }}
.filter-btn.active {{
    background:var(--tag-color, var(--primary));
    color:#fff !important;
    box-shadow:0 0 10px color-mix(in srgb, var(--tag-color) 40%, transparent);
}}
.btn-count {{ font-size:.7rem; opacity:.75; }}
.btn-clear {{
    background:transparent; border:1.5px solid var(--border2);
    color:var(--muted); font-family:inherit; font-size:.75rem;
    padding:.3rem .75rem; border-radius:9999px; cursor:pointer;
    transition:all .18s var(--ease); width:100%;
}}
.btn-clear:hover {{ border-color:var(--red); color:var(--red); }}

/* ── Bar chart ─────────────────────────────────────────────────────────── */
.bar-chart {{ display:flex; flex-direction:column; gap:.45rem; }}
.bar-row {{ display:grid; grid-template-columns:110px 1fr 36px; align-items:center; gap:.5rem; }}
.bar-label {{ font-size:.7rem; color:var(--muted2); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.bar-track {{ background:var(--surface2); border-radius:9999px; height:6px; overflow:hidden; }}
.bar-fill {{ height:100%; border-radius:9999px; transition:width .4s var(--ease); }}
.bar-count {{ font-size:.7rem; color:var(--muted); text-align:right; }}

/* ── Toolbar (search + count) ─────────────────────────────────────────── */
.toolbar {{
    display:flex; align-items:center; justify-content:space-between;
    gap:1rem; margin-bottom:1.5rem; flex-wrap:wrap;
}}
.search-box {{
    flex:1; min-width:220px; max-width:380px;
    display:flex; align-items:center; gap:.5rem;
    background:var(--surface); border:1px solid var(--border2);
    border-radius:9999px; padding:.45rem 1rem;
    transition:border-color .2s;
}}
.search-box:focus-within {{ border-color:var(--primary); }}
.search-box svg {{ flex-shrink:0; color:var(--muted); }}
.search-box input {{
    background:none; border:none; outline:none; color:var(--text);
    font-family:inherit; font-size:.875rem; width:100%;
}}
.search-box input::placeholder {{ color:var(--muted); }}
.result-count {{ font-size:.875rem; color:var(--muted); white-space:nowrap; }}
.active-count {{ color:var(--text); font-weight:600; }}

/* ── Gallery grid ─────────────────────────────────────────────────────── */
.gallery {{
    display:grid;
    grid-template-columns:repeat(auto-fill,minmax(240px,1fr));
    gap:1.25rem;
}}

/* ── Cards ────────────────────────────────────────────────────────────── */
.card {{
    background:var(--card-bg); border:1px solid var(--border);
    border-radius:var(--radius); overflow:hidden;
    box-shadow:var(--shadow-sm); display:flex; flex-direction:column;
    transition:transform .22s var(--ease), border-color .22s var(--ease), box-shadow .22s var(--ease);
}}
.card:hover {{ transform:translateY(-4px); border-color:var(--border2); box-shadow:var(--shadow-lg); }}
.card.hidden {{ display:none !important; }}

.card-image-container {{ position:relative; aspect-ratio:3/4; background:#06101f; overflow:hidden; }}
.card-image {{ width:100%; height:100%; object-fit:cover; transition:transform .3s var(--ease); display:block; }}
.card:hover .card-image {{ transform:scale(1.05); }}
.card-overlay {{ position:absolute; bottom:0; left:0; right:0; padding:.6rem .7rem; }}
.cat-badge {{
    display:inline-block; font-size:.68rem; font-weight:600;
    padding:.2rem .55rem; border-radius:.4rem;
    border:1px solid; backdrop-filter:blur(8px);
    background:rgba(0,0,0,.35);
}}

.card-content {{ padding:1rem; display:flex; flex-direction:column; flex:1; gap:.5rem; }}
.product-meta-row {{ display:flex; align-items:center; gap:.35rem; flex-wrap:wrap; min-height:1.5rem; }}
.product-id {{ font-size:.68rem; color:var(--muted); }}
.meta-pill {{
    font-size:.68rem; font-weight:500; padding:.15rem .45rem;
    border-radius:.3rem; white-space:nowrap;
}}
.brand-pill {{ background:rgba(99,102,241,.15); color:#a5b4fc; }}
.colour-pill {{ background:rgba(255,255,255,.07); color:var(--muted2); }}
.product-title {{
    font-size:.85rem; font-weight:600; line-height:1.35;
    display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
    overflow:hidden; min-height:2.3rem; color:var(--text);
}}
.chips {{ display:flex; flex-wrap:wrap; gap:.3rem; flex:1; align-content:flex-start; }}
.tag-chip {{
    font-size:.65rem; font-weight:600; padding:.18rem .5rem;
    border-radius:9999px; white-space:nowrap; letter-spacing:.02em;
    cursor:pointer; color:#fff; transition:filter .15s;
}}
.tag-chip:hover {{ filter:brightness(1.15); }}
.card-footer {{ padding-top:.65rem; border-top:1px solid var(--border); }}
.btn-link {{
    color:var(--muted); text-decoration:none; font-size:.78rem; font-weight:500;
    display:inline-flex; align-items:center; gap:.3rem; transition:color .15s;
}}
.btn-link:hover {{ color:var(--primary); }}
.icon {{ width:11px; height:11px; flex-shrink:0; }}

/* ── Empty state ──────────────────────────────────────────────────────── */
.empty-state {{
    grid-column:1/-1; text-align:center; padding:5rem 2rem;
    color:var(--muted); background:var(--surface);
    border:1px dashed var(--border); border-radius:var(--radius); display:none;
}}
.empty-state.visible {{ display:block; }}
.empty-state h3 {{ font-size:1.1rem; margin-bottom:.5rem; }}

/* ── Responsive ───────────────────────────────────────────────────────── */
@media(max-width:900px) {{
    .layout {{ grid-template-columns:1fr; }}
    .sidebar {{ position:static; height:auto; max-height:none; border-right:none; border-bottom:1px solid var(--border); }}
    .stats-grid {{ grid-template-columns:repeat(4,1fr); }}
}}
@media(max-width:540px) {{
    .stats-grid {{ grid-template-columns:repeat(2,1fr); }}
    .gallery {{ grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); }}
}}

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width:6px; }}
::-webkit-scrollbar-track {{ background:transparent; }}
::-webkit-scrollbar-thumb {{ background:var(--border2); border-radius:9999px; }}
</style>
</head>
<body>
<div class="layout">

  <!-- ── Sidebar ──────────────────────────────────────────────────────── -->
  <aside class="sidebar">
    <div class="logo">
            <h1>{dataset_label} Gallery</h1>
      <p>Curated dataset · human-worn images</p>
    </div>

    <div class="stats-grid">
      {stats_html}
    </div>

    <!-- Category distribution bar chart -->
    <div>
      <div class="filter-section-title">Category Distribution</div>
      <div class="bar-chart">
        {cat_bars_html}
      </div>
    </div>

    <!-- Category filter -->
    <div>
      <div class="filter-section-title">
        Filter by Category
        <button class="btn-clear" id="btn-clear-cat" style="width:auto;padding:.2rem .6rem;font-size:.7rem" onclick="clearCategoryFilters()">Clear</button>
      </div>
      <div class="filter-pills" id="cat-filter-pills">
        {cat_filter_btns}
      </div>
    </div>

    <!-- Tag filter -->
    <div>
      <div class="filter-section-title">
        Filter by Tag
        <button class="btn-clear" id="btn-clear-tag" style="width:auto;padding:.2rem .6rem;font-size:.7rem" onclick="clearTagFilters()">Clear</button>
      </div>
      <div class="filter-pills" id="tag-filter-pills">
        {tag_filter_btns}
      </div>
    </div>

    <button class="btn-clear" id="btn-clear-all" onclick="clearAllFilters()">Clear all filters</button>
  </aside>

  <!-- ── Main content ──────────────────────────────────────────────────── -->
  <div class="main">
    <div class="toolbar">
      <div class="search-box">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
        </svg>
        <input type="text" id="search-input" placeholder="Search by title, brand, colour, ID…" oninput="applyFilters()">
      </div>
      <div class="result-count">
        Showing <span class="active-count" id="visible-count">{len(df)}</span> of {len(df)} images
      </div>
    </div>

    <main class="gallery" id="gallery">
      {all_cards_html}
      <div class="empty-state" id="empty-state">
        <h3>No images match</h3>
        <p>Try adjusting your filters or search query.</p>
      </div>
    </main>
  </div>
</div>

<script>
const activeTags = new Set();
const activeCategories = new Set();
const cards = Array.from(document.querySelectorAll('.card'));
const visibleCount = document.getElementById('visible-count');
const emptyState = document.getElementById('empty-state');
const searchInput = document.getElementById('search-input');

// ── Category filter ────────────────────────────────────────────────────
document.getElementById('cat-filter-pills').addEventListener('click', e => {{
    const btn = e.target.closest('.cat-filter-btn');
    if (!btn) return;
    const cat = btn.dataset.category;
    if (activeCategories.has(cat)) {{
        activeCategories.delete(cat);
        btn.classList.remove('active');
    }} else {{
        activeCategories.add(cat);
        btn.classList.add('active');
    }}
    applyFilters();
}});

// ── Tag filter ─────────────────────────────────────────────────────────
document.getElementById('tag-filter-pills').addEventListener('click', e => {{
    const btn = e.target.closest('.tag-filter-btn');
    if (!btn) return;
    const tag = btn.dataset.tag;
    if (activeTags.has(tag)) {{
        activeTags.delete(tag);
        btn.classList.remove('active');
    }} else {{
        activeTags.add(tag);
        btn.classList.add('active');
    }}
    applyFilters();
}});

// ── Clicking a tag chip on a card activates that tag filter ────────────
document.getElementById('gallery').addEventListener('click', e => {{
    const chip = e.target.closest('.tag-chip');
    if (!chip) return;
    const tag = chip.dataset.tag;
    const btn = document.querySelector(`.tag-filter-btn[data-tag="${{tag}}"]`);
    if (btn) btn.click();
}});

// ── Filter logic ───────────────────────────────────────────────────────
function applyFilters() {{
    const query = searchInput.value.trim().toLowerCase();
    let visible = 0;
    cards.forEach(card => {{
        const cardTags = JSON.parse(card.dataset.tags || '[]');
        const cardCat = (card.dataset.categoryVal || '').toLowerCase();

        const tagOk = activeTags.size === 0 || [...activeTags].every(t => cardTags.includes(t));
        const catOk = activeCategories.size === 0 || activeCategories.has(card.dataset.categoryVal || '');

        let searchOk = true;
        if (query) {{
            const haystack = card.textContent.toLowerCase();
            searchOk = query.split(' ').every(term => haystack.includes(term));
        }}

        const show = tagOk && catOk && searchOk;
        card.classList.toggle('hidden', !show);
        if (show) visible++;
    }});
    visibleCount.textContent = visible;
    emptyState.classList.toggle('visible', visible === 0);
}}

function clearTagFilters() {{
    activeTags.clear();
    document.querySelectorAll('.tag-filter-btn.active').forEach(b => b.classList.remove('active'));
    applyFilters();
}}

function clearCategoryFilters() {{
    activeCategories.clear();
    document.querySelectorAll('.cat-filter-btn.active').forEach(b => b.classList.remove('active'));
    applyFilters();
}}

function clearAllFilters() {{
    clearTagFilters();
    clearCategoryFilters();
    searchInput.value = '';
    applyFilters();
}}
</script>
</body>
</html>
"""

    html_file = out_path / "visualization.html"
    html_file.write_text(html, encoding="utf-8")
    print(f"Visualization saved to: {html_file}")
    print(f"  {len(df)} images · {len(all_categories)} categories · {len(all_tags)} unique tags")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate curated dataset gallery")
    parser.add_argument("output_dir", help="Directory containing human_worn_sample.csv")
    parser.add_argument(
        "--pool-csv",
        default="",
        help="Path to the category sample pool CSV for brand/colour enrichment (optional)",
    )
    args = parser.parse_args()
    generate_html(args.output_dir, pool_csv=args.pool_csv or None)
