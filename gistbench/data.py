"""Data loading utilities for GISTBench."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd

from gistbench.schema import DATASET_CONFIGS, DatasetConfig

logger = logging.getLogger(__name__)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from CSV or JSON into a DataFrame.

    Expected columns: user_id, object_id, object_text, interaction_type, interaction_time
    """
    path = Path(path)

    if path.suffix == ".csv":
        df = pd.read_csv(path, dtype=str)
    elif path.suffix == ".json":
        df = pd.read_json(path, dtype=str)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, dtype=str, lines=True)
    elif path.is_dir():
        # Try to find a data file inside the directory
        for ext in [".csv", ".jsonl", ".json"]:
            candidates = list(path.glob(f"*{ext}"))
            if candidates:
                return load_dataset(candidates[0])
        raise FileNotFoundError(f"No data file found in {path}")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    required_cols = {"user_id", "object_id", "object_text", "interaction_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["user_id"] = df["user_id"].astype(str)
    df["object_id"] = df["object_id"].astype(str)

    logger.info(
        "Loaded %d engagements for %d users from %s",
        len(df),
        df["user_id"].nunique(),
        path,
    )
    return df


def detect_dataset_config(df: pd.DataFrame, dataset_name: str = "") -> DatasetConfig:
    """Detect or look up dataset configuration based on available signal types."""
    if dataset_name and dataset_name in DATASET_CONFIGS:
        return DATASET_CONFIGS[dataset_name]

    # Auto-detect from data
    types = set(df["interaction_type"].unique())
    return DatasetConfig(
        name=dataset_name or "custom",
        object_name="items",
        has_explicit_positive="explicit_positive" in types,
        has_implicit_positive="implicit_positive" in types,
        has_implicit_negative="implicit_negative" in types,
        has_explicit_negative="explicit_negative" in types,
    )


def validate_dataset(df: pd.DataFrame) -> None:
    """Assert dataset integrity at the start of a run.

    Checks required columns, non-empty data, valid interaction types,
    and that interaction_time (if present) has no missing values.
    """
    required_cols = {"user_id", "object_id", "object_text", "interaction_type"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if df.empty:
        raise ValueError("Dataset is empty")

    valid_types = {
        "explicit_positive", "implicit_positive",
        "implicit_negative", "explicit_negative",
    }
    bad_types = set(df["interaction_type"].unique()) - valid_types
    if bad_types:
        raise ValueError(f"Invalid interaction_type values: {bad_types}")

    if df["user_id"].isna().any():
        raise ValueError("user_id contains missing values")

    if df["object_id"].isna().any():
        raise ValueError("object_id contains missing values")

    if df["object_text"].isna().any():
        raise ValueError("object_text contains missing values")

    if "interaction_time" in df.columns:
        non_empty = df["interaction_time"].dropna()
        non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
        if len(non_empty) > 0 and len(non_empty) < len(df):
            logger.warning(
                "interaction_time has %d/%d missing values — sort order may be unreliable",
                len(df) - len(non_empty), len(df),
            )


def chunk_user_history(
    df: pd.DataFrame, user_id: str, chunk_size: int = 100
) -> list[pd.DataFrame]:
    """Split a user's history into chunks of at most chunk_size engagements.

    Sorted by interaction_time descending (most recent first).
    """
    user_df = df[df["user_id"] == user_id].copy()

    if "interaction_time" in user_df.columns:
        user_df = user_df.sort_values("interaction_time", ascending=False)

    chunks = []
    for i in range(0, len(user_df), chunk_size):
        chunks.append(user_df.iloc[i : i + chunk_size])

    return chunks


def sample_users(df: pd.DataFrame, n: int = 1000, seed: int = 42) -> list[str]:
    """Deterministically sample up to n users."""
    users = sorted(df["user_id"].unique())
    if len(users) <= n:
        return users

    rng = random.Random(seed)
    return sorted(rng.sample(users, n))
