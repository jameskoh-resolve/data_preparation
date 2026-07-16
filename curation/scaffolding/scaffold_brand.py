#!/usr/bin/env python
"""
Scaffold the directory structure for a new brand curation dataset.

Usage:
    python scripts/curation/scaffold_brand.py MyBrand
    python scripts/curation/scaffold_brand.py Myntra_India --brand-key myntra
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CURATED_DATASETS_DIR = PROJECT_ROOT / "curated_datasets"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


README_TEMPLATE = """\
# {brand_name} Curated Dataset

## Directory Structure

```
{brand_name}/
  curation/
    corrections/      # GT bounding box correction JSONs + scripts
    visualisation/    # Visualization outputs (HTML, JSON) + scripts
    html/             # Served HTML visualizations
    llm_anno/         # LLM annotation outputs
  categories/         # Category-level data
  v1/, v2/, ...       # Versioned dataset outputs (auto-generated)
```

## Workflow

### 1. Fix precision boxes (remove precision/fp_high_conf_incorrect issues)

```bash
python curated_datasets/{brand_name}/curation/corrections/fix_precision_boxes.py
```

### 2. Apply GT corrections

```bash
python curated_datasets/{brand_name}/curation/corrections/apply_gt_corrections.py gt_box_corrections-1.json
```

### 3. Generate visualization

```bash
python curated_datasets/{brand_name}/curation/visualisation/generate_gt_visualization.py
```

### 4. Generate versioned dataset + upload

```bash
python scripts/curation/generate_brand_dataset.py --brand {brand_key}

# Dry run (no upload):
python scripts/curation/generate_brand_dataset.py --brand {brand_key} --skip-upload
```

> **Warning**: Dry runs still create the `v{{N}}` directory. Delete it before running the
> final generation to avoid version skew. See `curated_datasets/README.md` for details.
"""


def scaffold(brand_name, brand_key):
    """Create the directory structure for a new brand."""
    brand_dir = CURATED_DATASETS_DIR / brand_name

    if brand_dir.exists():
        print(f"Error: Directory already exists: {brand_dir}")
        print(f"Remove it first if you want to re-scaffold.")
        sys.exit(1)

    # Create directory structure
    dirs = [
        brand_dir / "curation" / "corrections",
        brand_dir / "curation" / "visualisation",
        brand_dir / "curation" / "html",
        brand_dir / "curation" / "llm_anno",
        brand_dir / "categories",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        gitkeep = d / ".gitkeep"
        gitkeep.touch()

    # Copy template scripts
    corrections_dir = brand_dir / "curation" / "corrections"
    visualisation_dir = brand_dir / "curation" / "visualisation"
    html_dir = brand_dir / "curation" / "html"

    template_copies = [
        (TEMPLATES_DIR / "fix_precision_boxes.py", corrections_dir / "fix_precision_boxes.py"),
        (TEMPLATES_DIR / "apply_gt_corrections.py", corrections_dir / "apply_gt_corrections.py"),
        (TEMPLATES_DIR / "prepare_bbox_data.py", corrections_dir / "prepare_bbox_data.py"),
        (TEMPLATES_DIR / "generate_gt_visualization.py", visualisation_dir / "generate_gt_visualization.py"),
        (TEMPLATES_DIR / "bbox_adder.html", html_dir / "bbox_adder.html"),
    ]

    for src, dst in template_copies:
        shutil.copy2(src, dst)
        dst.chmod(0o755)

    # Generate README
    readme_content = README_TEMPLATE.format(brand_name=brand_name, brand_key=brand_key)
    readme_path = brand_dir / "README.md"
    readme_path.write_text(readme_content)

    print(f"Scaffolded new brand dataset: {brand_dir}")
    print(f"")
    print(f"Created:")
    for d in dirs:
        print(f"  {d.relative_to(PROJECT_ROOT)}/")
    for _, dst in template_copies:
        print(f"  {dst.relative_to(PROJECT_ROOT)}")
    print(f"  {readme_path.relative_to(PROJECT_ROOT)}")
    print(f"")
    print(f"Next steps:")
    print(f"  1. Place your raw curated CSV in: {brand_dir / 'curation'}/")
    print(f"  2. Run precision fix:  python {corrections_dir.relative_to(PROJECT_ROOT)}/fix_precision_boxes.py")
    print(f"  3. Apply corrections:  python {corrections_dir.relative_to(PROJECT_ROOT)}/apply_gt_corrections.py gt_box_corrections-1.json")
    print(f"  4. Generate dataset:   python scripts/curation/generate_brand_dataset.py --brand {brand_key}")


def main():
    parser = argparse.ArgumentParser(description="Scaffold directory structure for a new brand curation dataset.")
    parser.add_argument("brand_name", help="Brand directory name (e.g. 'Myntra_India', 'ASOS').")
    parser.add_argument(
        "--brand-key",
        default=None,
        help="Short brand key used in filenames (default: lowercase brand_name, e.g. 'myntra')."
    )
    args = parser.parse_args()

    brand_key = args.brand_key or args.brand_name.lower().split("_")[0]
    scaffold(args.brand_name, brand_key)


if __name__ == "__main__":
    main()
