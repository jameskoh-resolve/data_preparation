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

Before executing on an isolated GPU server, prepare image access or pre-download dataset images locally:

**Option A: Prepare Azure Blob Storage SAS URLs (Recommended)**
Uploads images to Azure Blob Storage and generates a SAS-enabled CSV in `azure_datasets/`:
```bash
PYTHONPATH=. .venv/bin/python curation/auto_annotate.py prep-azure configs/auto_annotate_westside_part_2.yaml
```

**Option B: Pre-download Images into Local Cache**
Pre-downloads image binaries directly into the local `cache/` directory:
```bash
PYTHONPATH=. .venv/bin/python curation/auto_annotate.py prefetch configs/auto_annotate_westside_part_2.yaml
```

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

---

### Prediction & Image Caching

- **Image Cache**: Downloaded images are cached locally as MD5-named `.jpg` files in `<output_dir>/cache/`. Subsequent runs re-use cached images without re-downloading them.
- **Prediction Cache**: Detector outputs (`locate_anything` / `fashion_model`) and LLM verification results (`gpt-4.1-mini`) are automatically cached in `<output_dir>/cache/predictions_cache.json`.
- **Resuming Runs**: If a pipeline run is interrupted or re-executed, all previously completed detections and LLM verification calls are instantly re-used from `predictions_cache.json` without incurring duplicate API costs or latency.
- **Cache Control**: Prediction caching is enabled by default (`use_prediction_cache: true`). To bypass caching for a fresh run, set `use_prediction_cache: false` under `dataset:` in your config YAML or delete `predictions_cache.json`.

---

## How It Works Under the Hood

- **Private Registry Lookup**: The `uv.toml` file is pre-configured to check the Visenze private registry (`https://pypi.fury.io/visenze/`) using your authenticated username token.
- **Self-contained pipeline**: All imports are absolute to the `data_preparation` repository. `vms` (`vi-multi-search`) dependencies are now fully localized, meaning the evaluation runs entirely from within this repository.

