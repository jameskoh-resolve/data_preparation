import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from datetime import datetime
from typing import ClassVar, NamedTuple, Type

import pandas as pd
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from loguru import logger
from path import Path
from pydantic import BaseModel
from tqdm import tqdm

"""
Annotation Cache Helpers
"""


class LoadCachedResult(NamedTuple):
    """Result of loading cached annotations."""

    annotated: pd.DataFrame
    pending: pd.DataFrame


class AnnotationCheckpoint:
    """Manages checkpoint persistence for annotation results with batched saves.

    Args:
        path: Directory to store checkpoint files.
        keys: Column names used as unique identifiers for annotations.
        save_interval: Number of records to accumulate before auto-saving.
    """

    def __init__(self, path: Path, keys: list[str], save_interval: int = 10):
        self.path = Path(path)
        self.keys = keys
        self.save_interval = save_interval
        self.file = self.path / "checkpoint.parquet"
        self._cache: pd.DataFrame | None = None
        self._pending: list[dict] = []
        self._session_records: list[dict] = []

    def _ensure_keys(self, df: pd.DataFrame, context: str) -> None:
        for key in self.keys:
            if key not in df.columns:
                raise ValueError(f"{context} is missing annotation key column: {key}")

    def _save_batch(self, records: list[dict]) -> None:
        """Save a batch of records to checkpoint file."""
        if not records:
            return

        df = pd.DataFrame(records)
        self._ensure_keys(df, "Annotation dataframe")

        self.path.makedirs_p()

        if not self.file.exists():
            df.to_parquet(self.file, index=False)
            self._cache = df
            return

        cached = self._cache if self._cache is not None else pd.read_parquet(self.file)
        self._ensure_keys(cached, str(self.file))
        merged_df = pd.concat([cached, df], axis=0)
        merged_df.to_parquet(self.file + ".tmp", index=False)
        os.replace(self.file + ".tmp", self.file)
        self._cache = merged_df

    def add(self, record: dict) -> None:
        """Add a record. Auto-saves when batch reaches save_interval."""
        self._session_records.append(record)
        self._pending.append(record)
        if len(self._pending) >= self.save_interval:
            self._save_batch(self._pending)
            self._pending = []

    def flush(self) -> None:
        """Save any remaining pending records and cleanup duplicates."""
        if self._pending:
            self._save_batch(self._pending)
            self._pending = []
        self._cleanup()
        if self._cache is not None:
            logger.info(f"Annotation checkpoint finalized at {self.file}")

    def _cleanup(self) -> None:
        """Remove duplicate entries, keeping the latest annotation based on order."""
        if not self.file.exists():
            return

        cached = pd.read_parquet(self.file)
        self._ensure_keys(cached, str(self.file))

        deduped = cached.drop_duplicates(subset=self.keys, keep="last")
        deduped.to_parquet(self.file, index=False)
        self._cache = deduped

    def load(self, df: pd.DataFrame) -> LoadCachedResult:
        """Load cached annotations and split input into annotated vs pending rows."""
        self._ensure_keys(df, "Annotation dataframe")

        if not self.file.exists():
            logger.info(f"No annotation checkpoint found at {self.file}, not loading annotated data")
            return LoadCachedResult(annotated=pd.DataFrame(), pending=df)

        cached = pd.read_parquet(self.file)
        self._ensure_keys(cached, str(self.file))

        logger.info(f"Annotation checkpoint found at {self.file}, filtering checkpoint by annotation_df keys")
        keys_df = df[self.keys].drop_duplicates()
        annotated_df = cached.merge(keys_df, on=self.keys, how="inner")
        logger.info(f"Loaded {len(annotated_df)} annotated rows from checkpoint")

        # Filter out already-annotated rows
        if len(annotated_df) == 0:
            pending_df = df
        else:
            annotated_index = annotated_df.set_index(self.keys).index
            pending_df = df[~df.set_index(self.keys).index.isin(annotated_index)]
        
        logger.info(f"{len(pending_df)} rows pending annotation after filtering with checkpoint")

        return LoadCachedResult(annotated=annotated_df, pending=pending_df)

    def get_all_records(self) -> pd.DataFrame:
        """Return records added during the current session (via add()) as a DataFrame."""
        return pd.DataFrame(self._session_records) if self._session_records else pd.DataFrame()


"""
Base Annotator Class
"""


def build_product_metadata(row: pd.Series) -> str:
    """Build comma-separated product metadata string from row fields."""
    colours_val = row.get("colors")
    colour = colours_val if isinstance(colours_val, str) and colours_val else row.get("colours")
    fields = [
        ("Title", row.get("title")),
        ("Description", row.get("description")),
        ("Gender", row.get("gender")),
        ("Colour", colour),
        ("Category", row.get("category")),
        ("Brand", row.get("brand")),
    ]
    return ", ".join(f"{key}: {val}" for key, val in fields if isinstance(val, str) and val)


class BaseAnnotator:
    annotation_output_fields: list[str] = []
    annotation_output_object: ClassVar[Type[BaseModel]]  # Subclasses must override
    prompt_path: Path | None = None  # Subclasses must override (or set before calling super().__init__)

    def __init__(
        self,
        system_path: Path,
        model_name: str = "gpt-4o-mini-2024-07-18",
        prod_type: str = "fashion",
    ):
        if system_path is None:
            raise ValueError("system_path is required")
        if self.prompt_path is None:
            raise ValueError("Subclasses of BaseAnnotator must set prompt_path before calling super().__init__()")

        self.model_name = model_name
        self.prod_type = prod_type

        logger.info(
            "Initializing {}: prompt={} model_name={} prod_type={}",
            self.__class__.__name__,
            self.prompt_path,
            model_name,
            prod_type,
        )

        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_path.read_text().strip()),
                HumanMessagePromptTemplate.from_template(self.prompt_path.read_text().strip()),
            ]
        )

    def process_row(self, row: pd.Series) -> dict:
        raise NotImplementedError("Subclasses must implement process_row method")


class LLMAnnotator:
    """Orchestrates parallel LLM annotation with automatic checkpointing.

    Args:
        annotator: BaseAnnotator instance that implements process_row.
        checkpoint_dir: Directory to store checkpoint files. If None, auto-generates
            as `.annotation_checkpoints/{AnnotatorClassName}_{HHMM_DDMMYY}`.
        checkpoint_interval: Number of annotations between checkpoint saves.
        max_workers: Maximum number of parallel threads.
        row_timeout: Timeout in seconds for each row annotation. None means no timeout.
    """

    def __init__(
        self,
        annotator: BaseAnnotator,
        checkpoint_dir: Path | None = None,
        checkpoint_interval: int = 10,
        max_workers: int = 8,
        row_timeout: float | None = 120.0,
    ):
        self.annotator = annotator
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.max_workers = max_workers
        self.row_timeout = row_timeout

    def _resolve_checkpoint_dir(self) -> Path:
        """Resolve checkpoint directory, auto-generating with timestamp if None."""
        if self.checkpoint_dir is not None:
            return Path(self.checkpoint_dir)
        timestamp = datetime.now().strftime("%H%M_%d%m%y")
        return Path(".annotation_checkpoints") / f"{self.annotator.__class__.__name__}_{timestamp}"

    def _build_record(self, row: pd.Series, result: dict, keys: list[str]) -> dict:
        """Build a record dict combining keys and annotation results."""
        record = {key: row[key] for key in keys}
        for field in self.annotator.annotation_output_fields:
            record[field] = result.get(field)
        return record

    def _merge_results(
        self,
        cached: pd.DataFrame,
        new_records: pd.DataFrame,
        keys: list[str],
    ) -> pd.DataFrame:
        """Merge cached and new annotations, dropping rows with missing output fields."""
        output_fields = self.annotator.annotation_output_fields

        if len(new_records) == 0:
            logger.warning("No rows were successfully annotated")
            return cached

        num_non_annotated = new_records[output_fields].isna().any(axis=1).sum()
        logger.info(f"Annotation complete with {len(new_records) - num_non_annotated} annotated rows, {num_non_annotated} non-annotated rows are dropped")
        new_records = new_records.dropna(subset=output_fields)

        merged = pd.concat([cached, new_records], axis=0)
        merged = merged.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)

        logger.info(f"Final annotated dataframe has {len(merged)} rows after merging with checkpoint data and dropping duplicates based on annotation keys: {keys}")
        return merged

    def annotate(self, df: pd.DataFrame, annotation_keys: list[str]) -> pd.DataFrame:
        """Annotate a dataframe using parallel processing with checkpointing.

        Args:
            df: DataFrame to annotate.
            annotation_keys: Columns used as unique keys for annotation tracking.

        Returns:
            DataFrame with annotation results.
        """
        checkpoint_dir = self._resolve_checkpoint_dir()
        checkpoint = AnnotationCheckpoint(
            path=checkpoint_dir,
            keys=annotation_keys,
            save_interval=self.checkpoint_interval,
        )

        load_result = checkpoint.load(df)

        if len(load_result.pending) == 0:
            logger.info("No rows to annotate, all rows are already annotated in cache")
            return load_result.annotated

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.annotator.process_row, row): (idx, row) for idx, row in load_result.pending.iterrows()}

            # Redirect loguru through tqdm.write to avoid fragmenting the progress bar
            logger.remove()
            tqdm_sink_id = logger.add(lambda msg: tqdm.write(msg.rstrip(), file=sys.stderr), colorize=True)

            for future in tqdm(as_completed(futures), total=len(futures), desc="Annotating"):
                idx, row = futures[future]
                try:
                    result = future.result(timeout=self.row_timeout)
                except FuturesTimeoutError:
                    logger.error(f"Row {idx} timed out after {self.row_timeout}s, skipping this row")
                    continue
                except Exception as exc:
                    logger.error(f"Row {idx} generated an exception: {exc}, skipping this row")
                    continue
                record = self._build_record(row, result, annotation_keys)
                checkpoint.add(record)

            # Restore default loguru sink
            logger.remove(tqdm_sink_id)
            logger.add(sys.stderr)

        checkpoint.flush()

        new_records = checkpoint.get_all_records()
        return self._merge_results(load_result.annotated, new_records, annotation_keys)


def annotate_df(
    df: pd.DataFrame,
    annotation_keys: list[str],
    annotator: BaseAnnotator,
    checkpoint_dir: Path | None = None,
    checkpoint_interval: int = 10,
    max_workers: int = 8,
    row_timeout: float | None = 120.0,
) -> pd.DataFrame:
    """Annotate a dataframe using the provided annotator with parallel processing.

    Wrapper around LLMAnnotator for backwards compatibility.

    Refers to `LLMAnnotator` for details of the parameters.
    """
    return LLMAnnotator(
        annotator=annotator,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        max_workers=max_workers,
        row_timeout=row_timeout,
    ).annotate(df, annotation_keys)
