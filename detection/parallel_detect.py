"""
ParallelDetect: multi-GPU parallel wrapper for ProductDetect / SimpleDetect.

Follows the same multiprocessing pattern as vms/embs/image_dense_emb.py:
- Top-level worker function receives a dict of args
- Each worker sets CUDA_VISIBLE_DEVICES before importing/creating the detector
- DataFrame is split into chunks, one per worker
- Workers write results to parquet buffers
- Parent concatenates and returns ordered results
"""

from __future__ import annotations

import os
import sys
import time
import tempfile
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import requests
from loguru import logger
from PIL import Image
from tqdm.auto import tqdm


IM_URL_COL = "im_url"
DETECTION_COL = "detections"


def _discover_nvidia_lib_dirs() -> List[str]:
    """Find nvidia runtime lib directories from current Python sys.path.

    Looks for paths like: <site-packages>/nvidia/*/lib
    """
    lib_dirs: List[str] = []
    seen = set()
    for p in sys.path:
        if not p:
            continue
        base = Path(p)
        nvidia_root = base / "nvidia"
        if not nvidia_root.exists() or not nvidia_root.is_dir():
            continue
        for lib_dir in nvidia_root.glob("*/lib"):
            if not lib_dir.exists() or not lib_dir.is_dir():
                continue
            lib_str = str(lib_dir)
            if lib_str in seen:
                continue
            seen.add(lib_str)
            lib_dirs.append(lib_str)
    return lib_dirs


def _ensure_nvidia_runtime_libs_in_ld_path() -> None:
    """Ensure NVIDIA runtime shared library paths are present in LD_LIBRARY_PATH."""
    lib_dirs = _discover_nvidia_lib_dirs()
    if not lib_dirs:
        return

    existing = [x for x in os.environ.get("LD_LIBRARY_PATH", "").split(":") if x]
    merged = []
    seen = set()

    # Prepend discovered dirs so process-local wheels override stale system libs.
    for path in lib_dirs + existing:
        if path in seen:
            continue
        seen.add(path)
        merged.append(path)

    os.environ["LD_LIBRARY_PATH"] = ":".join(merged)


def _load_image_rgb(source: Any) -> Optional[np.ndarray]:
    """Load an image source into RGB HWC uint8 ndarray.

    Supports: numpy array, http(s) URL string, local file path string.
    Returns None on failure.
    """
    try:
        if isinstance(source, np.ndarray):
            img = source
            if img.ndim == 2:
                img = np.stack([img, img, img], axis=2)
            elif img.ndim == 3 and img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            return img.astype(np.uint8) if img.dtype != np.uint8 else img

        if isinstance(source, (bytes, bytearray)):
            return np.array(Image.open(BytesIO(source)).convert("RGB"))

        if isinstance(source, str):
            if source.startswith(("http://", "https://")):
                from utils.vis_image import _get_image_request_headers
                headers = _get_image_request_headers(source)
                resp = requests.get(source, headers=headers, timeout=15)
                resp.raise_for_status()
                return np.array(Image.open(BytesIO(resp.content)).convert("RGB"))
            else:
                img = cv2.imread(source)
                if img is None:
                    return None
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        pass
    return None


def _detect_worker(args: Dict[str, Any]) -> None:
    """Worker function invoked in a child process via Pool.map.

    Pattern mirrors image_dense_emb.worker():
    - Receives all config in a single dict
    - Sets CUDA_VISIBLE_DEVICES before importing detector
    - Iterates over chunk rows with tqdm
    - Writes results to parquet buffer
    """
    chunk: pd.DataFrame = args["chunk"]
    model_dir: str = args["model_dir"]
    detector_type: str = args["detector_type"]
    worker_id: int = args["worker_id"]
    gpu: int = args["gpu"]
    buffer_path: str = args["buffer_path"]
    image_col: str = args["image_col"]
    box_concept: str = args["box_concept"]
    box_concept_col: Optional[str] = args["box_concept_col"]
    multiple_object: bool = args["multiple_object"]
    detection_limit: int = args["detection_limit"]
    category_overlap_thresh: float = args["category_overlap_thresh"]
    detector_kwargs: Dict[str, Any] = args["detector_kwargs"]

    logger.info("Worker {} starting on GPU {}...", worker_id, gpu)

    # Pin GPU before importing ONNX-based detector (same as image_dense_emb.py)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    _ensure_nvidia_runtime_libs_in_ld_path()

    # Support score threshold override without requiring detector __init__ args.
    score_thr_override = detector_kwargs.get("score_thr")
    init_kwargs = {k: v for k, v in detector_kwargs.items() if k != "score_thr"}

    # Import and instantiate detector inside worker process
    if detector_type == "simple":
        from detection.simple_detect import SimpleDetect
        detector = SimpleDetect(
            model_dir=model_dir,
            category_overlap_thresh=category_overlap_thresh,
            **init_kwargs,
        )
    elif detector_type == "rtmdet":
        from detection.rtmdet_detect import RTMDetDetect
        detector = RTMDetDetect(
            model_dir=model_dir,
            category_overlap_thresh=category_overlap_thresh,
            **init_kwargs,
        )
    else:
        from detection.centernet_detect import CenterNetDetect
        detector = CenterNetDetect(
            model_dir=model_dir,
            category_overlap_thresh=category_overlap_thresh,
            **init_kwargs,
        )

    if score_thr_override is not None and hasattr(detector, "_cfg") and hasattr(detector._cfg, "score_thr"):
        detector._cfg.score_thr = float(score_thr_override)

    # Run detection per row — access by column index since itertuples
    # renames columns starting with '_' (making them inaccessible by name).
    col_names = list(chunk.columns)
    image_col_idx = col_names.index(image_col)

    results: List[List[Dict[str, Any]]] = []
    for row in tqdm(
        chunk.itertuples(index=False),
        position=worker_id,
        total=len(chunk),
        mininterval=1.0,
    ):
        try:
            image = _load_image_rgb(row[image_col_idx])
            if image is None:
                results.append([])
                continue

            use_box_concept = box_concept
            if box_concept_col:
                bc_idx = col_names.index(box_concept_col)
                val = row[bc_idx]
                if val and str(val).strip():
                    use_box_concept = str(val).strip()

            dets = detector.detect(
                image,
                box_concept=use_box_concept,
                multiple_object=multiple_object,
                detection_limit=detection_limit,
            )
            results.append(dets)
        except Exception as exc:
            logger.warning("Worker {}: detection failed for one image: {}", worker_id, exc)
            results.append([])

    # Write only serializable output columns to parquet (never image ndarrays)
    out_df = pd.DataFrame({DETECTION_COL: results})
    out_df.to_parquet(buffer_path, engine="pyarrow")


class ParallelDetect:
    """Multi-GPU parallel detector following the image_dense_emb.py pattern.

    Usage:
        detector = ParallelDetect(
            model_dir="models/product_item_13_01",
            detector="simple",
            devices=["0", "1"],
            per_gpu=2,
        )
        # From a list of images (numpy arrays or URLs)
        results = detector.detect_images(images, box_concept="all")

        # From a DataFrame with an image column
        df_out = detector.detect_dataframe(df, image_col="im_url")
    """

    def __init__(
        self,
        model_dir: str,
        detector: str = "product",
        devices: Optional[List[str]] = None,
        per_gpu: int = 1,
        category_overlap_thresh: float = 0.3,
        detector_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_dir = str(model_dir)
        self.detector_type = self._normalize_detector(detector)
        self.per_gpu = max(int(per_gpu), 1)
        self.category_overlap_thresh = category_overlap_thresh
        self.detector_kwargs = detector_kwargs or {}

        # Resolve devices from env or argument
        if devices is None:
            env_gpus = os.environ.get("GPUS", "0")
            devices = [g.strip() for g in env_gpus.split(",") if g.strip()]
        self.devices = devices

    @staticmethod
    def _normalize_detector(name: str) -> str:
        n = name.strip().lower()
        if n in ("product", "productdetect", "centernet", "centernetdetect"):
            return "centernet"
        if n in ("simple", "simpledetect"):
            return "simple"
        if n in ("rtmdet", "rtmdetdetect"):
            return "rtmdet"
        raise ValueError(f"Unknown detector '{name}'. Use 'centernet', 'rtmdet', or 'simple'.")

    def detect_dataframe(
        self,
        dataframe: pd.DataFrame,
        image_col: str = IM_URL_COL,
        box_concept: str = "all",
        box_concept_col: Optional[str] = None,
        multiple_object: bool = True,
        detection_limit: int = -1,
    ) -> pd.DataFrame:
        """Run parallel detection over a DataFrame.

        Args:
            dataframe: Input DataFrame with image sources in image_col.
            image_col: Column containing image source (ndarray / URL / path).
            box_concept: Default category filter passed to detector.detect().
            box_concept_col: Optional per-row override column for box_concept.
            multiple_object: Enable multi-object output.
            detection_limit: Max detections per image (-1 = unlimited).

        Returns:
            Copy of dataframe with added 'detections' column (list of dicts).
        """
        if image_col not in dataframe.columns:
            raise ValueError(f"Column '{image_col}' not found in dataframe")
        if len(dataframe) == 0:
            out = dataframe.copy()
            out[DETECTION_COL] = pd.Series(dtype=object)
            return out

        start_time = time.perf_counter()

        # Build GPU worker list (same pattern as image_dense_emb._encode)
        gpus = [int(gpu) for gpu in self.devices]
        gpus *= self.per_gpu
        gpus = gpus[: min(len(gpus), len(dataframe))]
        logger.info("Per gpu with {} workers, total {} workers", self.per_gpu, len(gpus))

        # Split dataframe into chunks via index (avoids pandas FutureWarning)
        index_splits = np.array_split(np.arange(len(dataframe)), len(gpus))
        chunks = [dataframe.iloc[idxs].reset_index(drop=True) for idxs in index_splits]

        # Prepare parquet buffer paths
        tmp_dir = tempfile.mkdtemp(prefix="parallel_detect_")
        buffer_paths = [str(Path(tmp_dir) / f"detect_{i}.parquet") for i in range(len(chunks))]

        # Build worker args (same dict-style as image_dense_emb.worker)
        arg_list = [
            {
                "worker_id": i,
                "gpu": gpu,
                "model_dir": self.model_dir,
                "detector_type": self.detector_type,
                "chunk": chunks[i],
                "buffer_path": buffer_paths[i],
                "image_col": image_col,
                "box_concept": box_concept,
                "box_concept_col": box_concept_col,
                "multiple_object": multiple_object,
                "detection_limit": detection_limit,
                "category_overlap_thresh": self.category_overlap_thresh,
                "detector_kwargs": self.detector_kwargs,
            }
            for i, gpu in enumerate(gpus)
        ]

        logger.info(
            "Starting parallel detection with {} workers across devices {}",
            len(gpus),
            gpus,
        )

        # Process pool (same as image_dense_emb: Pool + map)
        p = Pool(len(gpus))
        try:
            p.map(_detect_worker, arg_list)
        finally:
            p.close()
            p.join()

        logger.info("Inference time: {:.4f} (s)", time.perf_counter() - start_time)

        # Reassemble results in original order
        logger.info("Generating final results...")
        detection_series = pd.concat(
            [pd.read_parquet(bp)[DETECTION_COL] for bp in buffer_paths],
            ignore_index=True,
        )

        # Clean up temp files
        for bp in buffer_paths:
            try:
                os.remove(bp)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

        result_df = dataframe.copy()
        result_df[DETECTION_COL] = detection_series.values
        logger.info("Operation finished.")
        return result_df

    def detect_images(
        self,
        images: List[Any],
        box_concept: str = "all",
        multiple_object: bool = True,
        detection_limit: int = -1,
    ) -> List[List[Dict[str, Any]]]:
        """Convenience: detect on a list of images (ndarrays/URLs/paths).

        Returns list of detection results aligned with input order.
        """
        df = pd.DataFrame({"image_source": images})
        out = self.detect_dataframe(
            df,
            image_col="image_source",
            box_concept=box_concept,
            multiple_object=multiple_object,
            detection_limit=detection_limit,
        )
        return out[DETECTION_COL].tolist()
