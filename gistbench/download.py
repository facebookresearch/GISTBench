"""Dataset download utilities.

Downloads GISTBench datasets from Hugging Face Hub.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from gistbench.schema import DATASET_CONFIGS

from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger(__name__)

HF_REPO_ID = "facebook/gistbench"

ASSETS_DIR = Path(__file__).parent / "assets"


def download_dataset(
    dataset_name: str = "synthetic",
    cache_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Download a GISTBench dataset from Hugging Face Hub.

    Raises on failure — no silent fallback. Use ``load_mock_dataset()``
    explicitly for testing.

    Args:
        dataset_name: Name of the dataset split to download
            (e.g., "synthetic", "kuairec", "mind").
        cache_dir: Local directory to cache downloaded files.
            Defaults to ~/.cache/gistbench.

    Returns:
        DataFrame with columns: user_id, object_id, object_text,
        interaction_type, interaction_time.
    """
    if cache_dir is None:
        cache_dir = Path.home() / ".cache" / "gistbench"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached_file = cache_dir / f"{dataset_name}.parquet"

    # Try cached file first
    if cached_file.exists():
        try:
            df = pd.read_parquet(cached_file, dtype_backend="numpy_nullable")
            logger.info("Loading cached dataset from %s", cached_file)
            return _coerce_dtypes(df)
        except Exception as e:
            logger.error("Corrupt cache at %s (%s), re-downloading.", cached_file, e)
            cached_file.unlink(missing_ok=True)

    # Download from Hugging Face Hub
    ds = hf_load_dataset(
        HF_REPO_ID,
        split="train",
        cache_dir=str(cache_dir),
    )
    logger.info("Downloaded dataset from Hugging Face: %s", HF_REPO_ID)
    df = ds.to_pandas()

    # Cache locally as parquet
    df.to_parquet(cached_file)
    return _coerce_dtypes(df)


def load_mock_dataset() -> pd.DataFrame:
    """Load the bundled mock dataset (3 users, 25 engagements) for testing.

    This must be requested explicitly — it is never used as a silent fallback.
    """
    mock_path = ASSETS_DIR / "mock_dataset.json"
    if not mock_path.exists():
        raise FileNotFoundError(f"Mock dataset not found at {mock_path}.")
    logger.info("Loading mock dataset from %s", mock_path)
    return pd.read_json(mock_path, dtype=str)


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure user_id and object_id are strings (HF dataset has int64)."""
    df["user_id"] = df["user_id"].astype(str)
    df["object_id"] = df["object_id"].astype(str)
    return df
