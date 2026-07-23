# Data Preparation Pipeline

This repository contains the dataset curation, cleaning, distribution generation, and splitting pipeline.

## Prerequisites

- **Python version:** `>= 3.10`
- **uv package manager:** (Recommended) Installing with `uv` ensures fast, reliable, and reproducible environment setups.

---

## Installation & Setup

### 1. Configure Private Gemfury Authentication
Private packages (such as `data-factory`) are hosted on Visenze's Gemfury registry.

1. Locate your `FURY_AUTH` token in `~/.dltk.config`.
2. Export it as an environment variable so `uv` can authenticate with the private index:
   ```bash
   export UV_INDEX_VISENZE_USERNAME=$(grep FURY_AUTH ~/.dltk.config | cut -d'=' -f2)
   ```

### 2. Create the Virtual Environment
Initialize a virtual environment using `uv`:
```bash
uv venv
source .venv/bin/activate
```

### 3. Install Dependencies
Install the required packages from `requirements.txt`:
```bash
uv pip install -r requirements.txt
```

---

## Auto-Annotate Curation Pipeline

The `curation/auto_annotate.py` script runs object detection (`locate_anything` / `fashion_model`), deduplication (IOU/IOMin overlap policy), and LLM crop verification (`gpt-4.1-mini`).

### 1. Local Preparation (Machine with Internet Access)

Before executing on an isolated GPU server, prepare image access by uploading images to Azure Blob Storage and generating a SAS-enabled CSV in `azure_datasets/`:
```bash
PYTHONPATH=. .venv/bin/python curation/auto_annotate.py prep-azure configs/auto_annotate_westside_part_2.yaml
```

`main` downloads and caches each image locally (in `<output_dir>/cache/image_cache/`) as it processes the dataset, so no separate prefetch step is needed.

---

### 2. Commit & Push Prepared Artifacts

Push the prepared CSV in `azure_datasets/` to GitHub so it is available on the Azure GPU server:
```bash
git add azure_datasets/
git commit -m "add prepared azure dataset CSV"
git push
```

---

### 3. Execution on Azure GPU Server

SSH into the Azure GPU server (where the `locate_anything` server is running locally on port 8080) and run:
```bash
git pull
PYTHONPATH=. .venv/bin/python curation/auto_annotate.py main configs/auto_annotate_westside_part_2.yaml
```

The script automatically detects the prepared CSV in `azure_datasets/` and outputs the curated dataset to `output/westside/`.

### 3b. Optional Two-Pass Workflow (Detection First, LLM Later)

Use this when you want to run detection-heavy work on Azure first, then run the LLM-heavy pass locally while reusing cache.

1. Run detection-only on Azure (fills detector cache keys in `detections_cache.json`):
```bash
PYTHONPATH=. .venv/bin/python curation/auto_annotate.py main configs/auto_annotate_westside_part_2.yaml --mode detection_only
```

2. Run full mode on local machine with the same config and cache path:
```bash
PYTHONPATH=. .venv/bin/python curation/auto_annotate.py main configs/auto_annotate_westside_part_2.yaml --mode full
```

Notes:
- Keep only one running process writing to the same cache files at a time.
- `detection_only` writes `<dataset>_detections_only.csv`.
- `full` writes `<dataset>_annotated.csv`.
- `output/.../cache/detections_cache.json`, `processed_cache.json`, and `llm_cache.json` are shared between both passes.

### 3c. Commands vs Modes (CLI Usage)

The auto-annotate CLI has two layers:

1. **Command layer** (subcommands)
- `prep-azure`: prepares/uploads images and writes Azure SAS CSV.
- `main`: runs the annotation pipeline (detection + dedup + LLM validation) and automatically (re)generates the CSV, `visualization.html`, `crops/`, and `crops.html` outputs at the end of every run — no separate command needed.

2. **Mode layer** (only used with `main`)
- `--mode full`: detection + dedup + LLM validation.
- `--mode detection_only`: detection + dedup only (skips CSV/HTML/crop generation).

Examples:
```bash
PYTHONPATH=. .venv/bin/python curation/auto_annotate.py prep-azure configs/auto_annotate_westside_part_2.yaml
PYTHONPATH=. .venv/bin/python curation/auto_annotate.py main configs/auto_annotate_westside_part_2.yaml --mode detection_only
PYTHONPATH=. .venv/bin/python curation/auto_annotate.py main configs/auto_annotate_westside_part_2.yaml --mode full
```

---

### Interactive HTML Visualization Gallery

Every `main` run automatically (re)generates two self-contained HTML galleries in your output directory, purely from cache if the images/detections/LLM results are already cached (no re-running inference or API calls needed):

- **`visualization.html`**: the full dataset gallery.
  - **Toggle `LLM Filtered`**: Easily toggle between showing **final LLM-validated boxes** and **raw model candidate boxes** (with LLM-rejected boxes highlighted via red dashed borders).
  - **Search & Filter**: Filter by class concept, LLM status (validated vs rejected), image ID, or prompt reason.
  - **Detail Modal**: Click any image card to open a full-screen inspector with class tags, box coordinates, and LLM explanation reasons.
- **`crops.html`**: a gallery of the exact crops sent to the LLM, alongside its validation result and reason, for reviewing LLM verification quality.

---

### Interactive Review GUI

For manually correcting/extending detections (approve, reject, hide/show, or draw new boxes), use the browser-based review tool:
```bash
PYTHONPATH=. .venv/bin/python curation/review_gui.py main configs/auto_annotate_westside_part_2.yaml
PYTHONPATH=. .venv/bin/python curation/review_gui.py main configs/auto_annotate_westside_part_2.yaml --port 7654
```

- **Grid (batch) view**: shows as many image thumbnails at once as fit your screen resolution (e.g. ~12 on a typical laptop display), each with its bounding boxes drawn. Click a thumbnail to open the detail view for that image.
- **Colors**: green = valid, red = invalid (LLM rejected, including blurry-flagged boxes), violet = new (human-added). Invalid boxes are **hidden by default** on both the grid and detail view.
- **Hotkeys**: `A`/`D` (or `←`/`→`) prev/next batch (grid view) or prev/next image (detail view), `Space` show/hide **all** invalid boxes at once (grid + detail view), `Enter` approve selected box, `X` reject selected box, `Esc` back to grid view.
- Individual boxes are shown/hidden by clicking their **Hide/Show** button in the detail view's box list (not via keyboard).
- **Click-drag** on the canvas (detail view) to draw a new box (select a class from `classes.txt` in the right panel first).
- Decisions are saved to `<cache_dir>/human_review.json`. **Saving alone does not change any dataset CSV** — run the `export` command below to apply your decisions.

Once you're done reviewing, apply your decisions to produce a final corrected CSV (this does not modify the original `{dataset}_annotated.csv`):
```bash
PYTHONPATH=. .venv/bin/python curation/review_gui.py export configs/auto_annotate_westside_part_2.yaml
```
This writes `{dataset}_human_reviewed.csv` to your output directory. Per box: an explicit **reject** excludes it, an explicit **approve** includes it (overriding the LLM verdict), and boxes with no explicit decision are included iff the LLM (or lack of LLM validation) already marked them valid. Hiding a box in the review UI is only a visibility toggle and does not exclude it. Boxes you drew manually are always included.


---

### Prediction & Image Caching

- **Image Cache**: Downloaded images are cached locally as MD5-named `.jpg` files in `<output_dir>/cache/image_cache/`. Subsequent runs re-use cached images without re-downloading them.
- **Prediction Cache**: Cached separately per pipeline stage in `<output_dir>/cache/`:
  - `detections_cache.json` — raw detector outputs (`locate_anything` / `fashion_model`).
  - `processed_cache.json` — post-dedup, post-enforce-area detections (the GPU1 → GPU2 handoff for the two-pass workflow).
  - `llm_cache.json` — LLM verification results (`is_valid` + `reason`) per box.
- **Resuming Runs**: If a pipeline run is interrupted or re-executed, all previously completed detections and LLM verification calls are instantly re-used from cache without incurring duplicate API costs or latency.
- **Cache Control**: Cache reuse is enabled by default (`reuse_cache: true`). To bypass caching for a fresh run, set `reuse_cache: false` under `dataset:` in your config YAML or delete the relevant cache files (`detections_cache.json`, `processed_cache.json`, `llm_cache.json`) from the cache directory.

---

## How It Works Under the Hood

- **Private Registry Lookup**: The `uv.toml` file is pre-configured to check the Visenze private registry (`https://pypi.fury.io/visenze/`) using your authenticated username token.
- **Self-contained pipeline**: All imports are absolute to the `data_preparation` repository. `vms` (`vi-multi-search`) dependencies are now fully localized, meaning the evaluation runs entirely from within this repository.

