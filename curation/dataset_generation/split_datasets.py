#!/usr/bin/env python
"""Split asos_dataset_v1 and cos_dataset_v1 into train/val/test CSVs (80/10/10).

Uses multilabel-aware stratification on concepts field to preserve concept distribution.

Outputs:
    curated_datasets/ASOS/v1/asos_dataset_v1_train.csv
    curated_datasets/ASOS/v1/asos_dataset_v1_val.csv
    curated_datasets/ASOS/v1/asos_dataset_v1_test.csv
    curated_datasets/ASOS/v1/asos_dataset_v1_split_diagnostics.csv
    curated_datasets/COS/v1/cos_dataset_v1_train.csv
    curated_datasets/COS/v1/cos_dataset_v1_val.csv
    curated_datasets/COS/v1/cos_dataset_v1_test.csv
    curated_datasets/COS/v1/cos_dataset_v1_split_diagnostics.csv
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import typer
from loguru import logger
from path import Path
from sklearn.model_selection import train_test_split

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    HAS_ITERSTRAT = True
except ImportError:
    HAS_ITERSTRAT = False

app = typer.Typer(pretty_exceptions_show_locals=False)

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 42


def parse_concepts(concepts_str: str) -> list[str]:
    """Parse comma-separated concepts string into deduplicated list of tokens.

    Args:
        concepts_str: Comma-separated concepts (e.g., "earring,bottom,top,shoe,ring").

    Returns:
        Deduplicated and trimmed list of concepts.
    """
    if not concepts_str or not isinstance(concepts_str, str):
        return []
    tokens = [t.strip() for t in concepts_str.split(",")]
    return list(dict.fromkeys([t for t in tokens if t]))  # Deduplicate while preserving order


def build_multilabel_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build multilabel indicator matrix from concepts column.

    Args:
        df: DataFrame with 'concepts' column.

    Returns:
        Tuple of (indicator matrix of shape [n_samples, n_classes], sorted list of unique concepts).
    """
    all_concepts = set()
    parsed_concepts = []

    for concepts_str in df["concepts"]:
        tokens = parse_concepts(concepts_str)
        parsed_concepts.append(tokens)
        all_concepts.update(tokens)

    concept_list = sorted(all_concepts)
    n_samples = len(df)
    n_concepts = len(concept_list)
    concept_to_idx = {c: i for i, c in enumerate(concept_list)}

    Y = np.zeros((n_samples, n_concepts), dtype=np.int32)
    for i, tokens in enumerate(parsed_concepts):
        for token in tokens:
            Y[i, concept_to_idx[token]] = 1

    return Y, concept_list


def compute_concept_prevalence(Y: np.ndarray, concept_list: list[str], indices: np.ndarray | None = None) -> dict[str, float]:
    """Compute per-concept prevalence as fraction of samples.

    Args:
        Y: Multilabel indicator matrix.
        concept_list: Sorted list of concept names.
        indices: Optional array of row indices to filter Y; if None, uses all rows.

    Returns:
        Dictionary mapping concept name to prevalence (0-1).
    """
    if indices is not None:
        Y_subset = Y[indices]
    else:
        Y_subset = Y

    prevalence = {}
    for i, concept in enumerate(concept_list):
        prevalence[concept] = Y_subset[:, i].sum() / len(Y_subset) if len(Y_subset) > 0 else 0.0
    return prevalence


def split_dataset(input_path: Path, name: str, output_dir: Path) -> None:
    """Split dataset into train/val/test using multilabel stratification.

    Args:
        input_path: Path to input CSV.
        name: Dataset name (for logging and output file prefix).
        output_dir: Output directory for train/val/test files and diagnostics.
    """
    if not HAS_ITERSTRAT:
        logger.error(
            "iterative-stratification not found. Install with: pip install iterative-stratification"
        )
        sys.exit(1)

    df = pd.read_csv(input_path)
    logger.info(f"{name}: {len(df)} total rows")

    # Build multilabel indicator matrix
    Y, concept_list = build_multilabel_matrix(df)
    logger.info(f"  detected {len(concept_list)} unique concepts")

    # Log rare concept warnings
    concept_counts = Y.sum(axis=0)
    rare_concepts = [c for c, cnt in zip(concept_list, concept_counts) if cnt < 3]
    if rare_concepts:
        logger.warning(f"  rare concepts (support < 3): {rare_concepts}")

    # Stage 1: split 80/20 (train vs val+test)
    splitter_1 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )
    train_idx, val_test_idx = next(splitter_1.split(df, Y))

    # Stage 2: split val+test 50/50 (val vs test), which gives 10/10 of original
    Y_val_test = Y[val_test_idx]
    splitter_2 = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=RANDOM_SEED,
    )
    val_idx_local, test_idx_local = next(splitter_2.split(df.iloc[val_test_idx], Y_val_test))
    val_idx = val_test_idx[val_idx_local]
    test_idx = val_test_idx[test_idx_local]

    # Extract splits
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    splits = {"train": train_df, "val": val_df, "test": test_df}

    # Compute prevalence for diagnostics
    full_prevalence = compute_concept_prevalence(Y, concept_list)
    train_prevalence = compute_concept_prevalence(Y, concept_list, train_idx)
    val_prevalence = compute_concept_prevalence(Y, concept_list, val_idx)
    test_prevalence = compute_concept_prevalence(Y, concept_list, test_idx)

    # Write split files
    for split_name, split_df in splits.items():
        out_path = output_dir / f"{name}_{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        logger.info(f"  {split_name}: {len(split_df)} rows -> {out_path}")

    # Compute and log split quality metrics
    total = len(train_df) + len(val_df) + len(test_df)
    train_pct = len(train_df) / total * 100
    val_pct = len(val_df) / total * 100
    test_pct = len(test_df) / total * 100

    logger.info(
        f"  split ratios: train={train_pct:.1f}%  val={val_pct:.1f}%  test={test_pct:.1f}%"
    )

    # Compute max prevalence drift per concept
    max_drift = 0.0
    for concept in concept_list:
        drifts = [
            abs(train_prevalence[concept] - full_prevalence[concept]),
            abs(val_prevalence[concept] - full_prevalence[concept]),
            abs(test_prevalence[concept] - full_prevalence[concept]),
        ]
        max_drift = max(max_drift, max(drifts))

    logger.info(f"  max concept prevalence drift: {max_drift:.3f}")

    # Write diagnostics CSV
    diag_data = []
    for concept in concept_list:
        diag_data.append({
            "concept": concept,
            "full": full_prevalence[concept],
            "train": train_prevalence[concept],
            "val": val_prevalence[concept],
            "test": test_prevalence[concept],
            "train_drift": abs(train_prevalence[concept] - full_prevalence[concept]),
            "val_drift": abs(val_prevalence[concept] - full_prevalence[concept]),
            "test_drift": abs(test_prevalence[concept] - full_prevalence[concept]),
        })

    diag_df = pd.DataFrame(diag_data)
    diag_path = output_dir / f"{name}_split_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False)
    logger.info(f"  split diagnostics -> {diag_path}")


@app.command()
def main(
    input_path: str = typer.Argument(..., help="Path to input CSV to split"),
    name: str = typer.Option(None, "--name", "-n", help="Dataset name for output file prefix"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory"),
) -> None:
    inp = Path(input_path)
    n = name if name else inp.stem
    out = Path(output_dir) if output_dir else inp.parent
    split_dataset(inp, n, out)
    logger.info("Done.")


if __name__ == "__main__":
    app()
