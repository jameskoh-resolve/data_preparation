import os
import yaml

TARGETS = {
    "lyst": 3000,
    "aijo_luxe": 1500,
    "asos": 1500,
    "veromoda": 150,
    "cos": 1500,
    "selectedhomme": 100,
    "westside": 250,
}

RELEVANT_CLASSES = [
    "belt", "ring", "gloves", "hair_accessories", "scarf", "tie",
    "earring", "bracelet", "necklace", "watch", "headwear", "bag", "eyewear", "innertop"
]

for catalog, target in TARGETS.items():
    if catalog in ["lyst", "aijo_luxe", "asos", "veromoda"]:
        csv_path = f"curated_datasets/combined_catalogs/catalog_dataset_part_1/{catalog}/category_sample_pool.csv"
        image_col = "im_url"
    else:
        csv_path = f"catalogs/{catalog}_catalog.csv"
        image_col = "main_image_url"

    config = {
        "run": {
            "target_size": target,
            "catalog_name": catalog,
            "seed": 42,
            "output_dir": f"curated_datasets/combined_catalogs/catalog_dataset_part_1/{catalog}",
            "batch_size": 2000
        },
        "catalog": {
            "csv": csv_path,
            "image_url_column": image_col
        },
        "dedup": {
            "exclude_urls_csv": "curated_datasets/combined_catalogs/v4/accessory_detect_v4_dataset.csv",
            "exclude_hash_column": "im_name"
        },
        "tagging": {
            "model_config": "configs/tagging_models/fashion_eval_image.yaml",
            "n_jobs": 16
        },
        "llm": {
            "model": "gpt-4.1-mini-2025-04-14",
            "workers": 8
        },
        "classes": {
            "relevant_classes": RELEVANT_CLASSES
        }
    }
    
    out_path = f"configs/catalog_dataset_part_1/{catalog}_class_sampler.yaml"
    with open(out_path, "w") as f:
        yaml.dump(config, f, sort_keys=False)
    print(f"Created {out_path}")

