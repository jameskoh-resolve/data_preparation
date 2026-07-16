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

## How It Works Under the Hood

- **Private Registry Lookup**: The `uv.toml` file is pre-configured to check the Visenze private registry (`https://pypi.fury.io/visenze/`) using your authenticated username token.
- **Self-contained pipeline**: All imports are absolute to the `data_preparation` repository. `vms` (`vi-multi-search`) dependencies are now fully localized, meaning the evaluation runs entirely from within this repository.
