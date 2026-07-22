#!/usr/bin/env python3
"""Retrieve a product catalog via Data Factory and save it as CSV.

Edit the configuration values below, then run:
    python datasets/catalogs/retrieve_catalog.py
"""

from __future__ import annotations

import importlib

from path import Path


# Update these values directly before running the script.
CATALOG_NAME = "ASOS"
CATALOG_ID = "10"
ACCESS_KEY = "e209f2d10fad67108fa5c0373cc69ce40796a3da"
SECRET_KEY = "bd724fd328f00d2612ff22fcba2f9b0cedd6f3ad"
OUTPUT_CSV = None  # e.g. "datasets/catalogs/my_catalog.csv"


def main() -> None:
    if not ACCESS_KEY or not SECRET_KEY:
        raise ValueError(
            "Missing credentials. Set ACCESS_KEY and SECRET_KEY in this script."
        )

    default_output = Path(__file__).dirname() / f"{CATALOG_NAME}_catalog.csv"
    output_path = Path(OUTPUT_CSV) if OUTPUT_CSV else default_output

    product_catalog = importlib.import_module("data_factory.client.product_catalog")
    catalog_get_data_cls = getattr(product_catalog, "ProductCatalogGetData")

    catalog = catalog_get_data_cls(CATALOG_ID, ACCESS_KEY, SECRET_KEY)
    df = catalog.read_dataframe()

    output_path.parent.makedirs_p()
    df.to_csv(str(output_path), index=False)

    print(f"Saved {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
