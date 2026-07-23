#!/usr/bin/env python
"""Standalone gallery visualizer for a flat catalog CSV (im_url + metadata, no detections).

Usage:
    PYTHONPATH=. .venv/bin/python curation/visualize_catalog_sample.py \
        curated_datasets/combined_catalogs/catalog_dataset_part_1/combined.csv
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer

app = typer.Typer(pretty_exceptions_show_locals=False)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>__TITLE__</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
       background: #0f172a; color: #e2e8f0; min-height: 100vh; }
.header { position: sticky; top: 0; z-index: 100; background: #1e293b; border-bottom: 1px solid #334155;
          padding: 12px 20px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
          box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
.header h1 { font-size: 16px; white-space: nowrap; color: #f8fafc; font-weight: 700; }
.header .stats { font-size: 12px; color: #94a3b8; background: #0f172a; padding: 4px 10px; border-radius: 20px;
                  border: 1px solid #334155; }
.controls { display: flex; align-items: center; gap: 10px; margin-left: auto; flex-wrap: wrap; }
.controls input[type="text"] { background: #334155; border: 1px solid #475569; border-radius: 6px; padding: 6px 12px;
                                color: #e2e8f0; font-size: 12px; width: 240px; outline: none; }
.controls select { background: #334155; border: 1px solid #475569; border-radius: 6px; padding: 6px 12px;
                    color: #e2e8f0; font-size: 12px; outline: none; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; padding: 16px; }
.card { background: #1e293b; border: 1px solid #334155; border-radius: 8px; overflow: hidden; cursor: pointer;
        transition: transform 0.1s, border-color 0.1s; }
.card:hover { transform: translateY(-2px); border-color: #64748b; }
.card img { width: 100%; height: 220px; object-fit: cover; display: block; background: #0f172a; }
.card .meta { padding: 8px 10px; font-size: 11px; color: #cbd5e1; }
.card .meta .catalog { display: inline-block; background: #334155; color: #93c5fd; border-radius: 4px;
                        padding: 1px 6px; font-size: 10px; margin-bottom: 4px; }
.card .meta .title { color: #f1f5f9; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.card .meta .sub { color: #94a3b8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.modal { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85); z-index: 1000;
         align-items: center; justify-content: center; }
.modal.active { display: flex; }
.modal-content { background: #1e293b; border-radius: 10px; max-width: 90vw; max-height: 90vh; overflow: auto;
                  display: flex; flex-direction: column; }
.modal-content img { max-width: 80vw; max-height: 70vh; object-fit: contain; display: block; margin: 0 auto; }
.modal-meta { padding: 16px; font-size: 13px; color: #cbd5e1; line-height: 1.6; }
.modal-close { position: absolute; top: 16px; right: 24px; font-size: 28px; color: #e2e8f0; cursor: pointer; }
</style>
</head>
<body>
<div class="header">
  <h1>__TITLE__</h1>
  <div class="stats" id="stats"></div>
  <div class="controls">
    <select id="catalogFilter"><option value="">All catalogs</option></select>
    <input type="text" id="search" placeholder="Search title / brand / category...">
  </div>
</div>
<div class="grid" id="grid"></div>
<div class="modal" id="modal" onclick="if(event.target===this) closeModal()">
  <span class="modal-close" onclick="closeModal()">&times;</span>
  <div class="modal-content">
    <img id="modalImg">
    <div class="modal-meta" id="modalMeta"></div>
  </div>
</div>
<script>
const data = __DATA_JSON__;
let filtered = data;

function render() {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  const frag = document.createDocumentFragment();
  filtered.forEach((item, i) => {
    const card = document.createElement('div');
    card.className = 'card';
    card.onclick = () => openModal(i);
    card.innerHTML = `
      <img src="${item.im_url}" loading="lazy" onerror="this.onerror=null;this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22><rect width=%22100%22 height=%22100%22 fill=%22%231e293b%22/><text x=%2250%25%22 y=%2250%25%22 fill=%22%2364748b%22 text-anchor=%22middle%22 font-size=%2210%22>Image Load Error</text></svg>'">
      <div class="meta">
        <span class="catalog">${item.catalog_name || ''}</span>
        <div class="title">${item.title || '(no title)'}</div>
        <div class="sub">${item.brand || ''} · ${item.category || ''}</div>
      </div>`;
    frag.appendChild(card);
  });
  grid.appendChild(frag);
  document.getElementById('stats').textContent = `${filtered.length} / ${data.length} images`;
}

function applyFilters() {
  const cat = document.getElementById('catalogFilter').value;
  const q = document.getElementById('search').value.toLowerCase();
  filtered = data.filter(item => {
    if (cat && item.catalog_name !== cat) return false;
    if (!q) return true;
    return [item.title, item.brand, item.category].some(v => (v || '').toLowerCase().includes(q));
  });
  render();
}

function openModal(i) {
  const item = filtered[i];
  document.getElementById('modalImg').src = item.im_url;
  document.getElementById('modalMeta').innerHTML = `
    <div><b>catalog:</b> ${item.catalog_name || ''}</div>
    <div><b>product_id:</b> ${item.product_id || ''}</div>
    <div><b>title:</b> ${item.title || ''}</div>
    <div><b>brand:</b> ${item.brand || ''}</div>
    <div><b>colour:</b> ${item.colour || ''}</div>
    <div><b>category:</b> ${item.category || ''}</div>
    <div><a href="${item.im_url}" target="_blank" style="color:#60a5fa;">Open Image Link &#8599;</a></div>`;
  document.getElementById('modal').classList.add('active');
}
function closeModal() { document.getElementById('modal').classList.remove('active'); }

function init() {
  const catalogs = [...new Set(data.map(d => d.catalog_name))].sort();
  const sel = document.getElementById('catalogFilter');
  catalogs.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c; opt.textContent = c;
    sel.appendChild(opt);
  });
  sel.onchange = applyFilters;
  document.getElementById('search').oninput = applyFilters;
  render();
}
window.onload = init;
</script>
</body>
</html>
"""


@app.command()
def main(
    csv_path: str = typer.Argument(..., help="Flat CSV with im_url + metadata columns"),
    output: str = typer.Option(None, help="Output HTML path (defaults next to the CSV)"),
    sample: int = typer.Option(0, help="If > 0, randomly sample this many rows instead of using all rows"),
    seed: int = typer.Option(0, help="Random seed used when --sample is set"),
):
    path = Path(csv_path)
    df = pd.read_csv(path)
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=seed)

    cols = [c for c in ("catalog_name", "product_id", "title", "brand", "colour", "category", "im_url") if c in df.columns]
    items = df[cols].fillna("").to_dict(orient="records")

    output_path = Path(output) if output else path.with_name(path.stem + "_visualization.html")
    html = (
        HTML_TEMPLATE.replace("__TITLE__", f"Catalog Sample — {path.stem} ({len(items)} images)")
        .replace("__DATA_JSON__", json.dumps(items))
    )
    output_path.write_text(html, encoding="utf-8")
    print(f"Wrote {output_path} ({len(items)} images)")


if __name__ == "__main__":
    app()
