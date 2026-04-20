"""Core data types for GISTBench."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gistbench.store import ResultsStore

logger = logging.getLogger(__name__)


@dataclass
class Engagement:
    """A single user-object engagement record."""

    user_id: str
    object_id: str
    object_text: str
    interaction_type: str  # explicit_positive, implicit_positive, implicit_negative, explicit_negative
    interaction_time: str = ""


@dataclass
class Interest:
    """A predicted interest with cited evidence objects."""

    name: str
    item_ids: list[str] = field(default_factory=list)
    evidence_excerpt: str = ""
    brief_rationale: str = ""



@dataclass
class IGResult:
    """Interest Groundedness result for one interest."""

    user_id: str
    dataset: str
    model: str
    interest: str
    verified: bool = False


@dataclass
class ISResult:
    """Interest Specificity result for one interest."""

    user_id: str
    dataset: str
    model: str
    interest: str
    correct: int = 0
    selected: int = 0
    backing: int = 0


@dataclass
class UserScore:
    """Final per-user GISTBench score."""

    user_id: str
    dataset: str
    model: str
    ig_normalized: float = 0.0
    is_normalized: float = 0.0
    harmonic_mean: float = 0.0
    oracle_count: int = 0


@dataclass
class Oracle:
    """Per-user oracle: the set of interest category IDs any model could verify.

    The oracle count is the denominator for IG Recall.  It must be computed
    across all evaluated models (or provided as ground truth) — using a
    single model's verified count makes IG_R = IG_P, which is meaningless.

    All entries are ``category_id`` (int), not names.
    """

    category_ids: dict[str, list[int]]  # user_id -> list of oracle category IDs

    def count(self, user_id: str) -> int:
        """Number of oracle categories for a user."""
        return len(self.category_ids.get(user_id, []))

    def user_ids(self) -> list[str]:
        """All user IDs with oracle data."""
        return sorted(self.category_ids.keys())

    @classmethod
    def from_file(cls, path: str | Path) -> "Oracle":
        """Load oracle from a JSON file.

        Expected format::

            {
              "oracle": {
                "user_1": [16, 133, 42, ...],
                "user_2": [201, 268, ...]
              }
            }

        Values are category IDs (int).
        """
        data = json.loads(Path(path).read_text())
        cat_ids: dict[str, list[int]] = {}
        for uid, ids in data["oracle"].items():
            cat_ids[uid] = [int(x) for x in ids]
        return cls(category_ids=cat_ids)

    def to_file(self, path: str | Path, dataset: str = "") -> None:
        """Write oracle to JSON in the format read by ``from_file``.

        ``dataset`` is included as a top-level key when non-empty so the
        snapshot is self-describing.
        """
        payload: dict[str, object] = {}
        if dataset:
            payload["dataset"] = dataset
        payload["oracle"] = self.category_ids
        Path(path).write_text(json.dumps(payload, indent=2))

    def merge(self, other: "Oracle") -> "Oracle":
        """Union per-user category IDs with another oracle.

        For each user present in either oracle, the resulting list is the
        sorted union of the two category ID sets.
        """
        merged: dict[str, list[int]] = {}
        for uid in set(self.category_ids) | set(other.category_ids):
            ids = set(self.category_ids.get(uid, [])) | set(
                other.category_ids.get(uid, [])
            )
            merged[uid] = sorted(ids)
        return Oracle(category_ids=merged)

    @classmethod
    def from_results_store(cls, store: "ResultsStore", dataset: str) -> "Oracle":
        """Compute oracle from a ResultsStore (cross-model).

        Uses the store's public ``compute_oracle_categories()`` method.
        Unmapped interests are dropped — only mapped interests count.
        """
        users = store.list_users(dataset)
        cat_ids: dict[str, list[int]] = {}
        for user_id in users:
            ids = store.compute_oracle_categories(user_id, dataset)
            if ids:
                cat_ids[user_id] = ids
        return cls(category_ids=cat_ids)


_BUNDLED_ORACLE = Path(__file__).parent / "assets" / "oracle_synthetic.json"
_MOCK_ORACLE = Path(__file__).parent / "assets" / "mock_oracle.json"


def load_bundled_oracle() -> Oracle:
    """Load the bundled oracle for the synthetic dataset (997 users)."""
    if _BUNDLED_ORACLE.exists():
        return Oracle.from_file(_BUNDLED_ORACLE)
    raise FileNotFoundError(
        f"Bundled oracle not found at {_BUNDLED_ORACLE}."
    )


def load_mock_oracle() -> Oracle:
    """Load the small mock oracle (3 users) for tests."""
    if _MOCK_ORACLE.exists():
        return Oracle.from_file(_MOCK_ORACLE)
    raise FileNotFoundError(
        f"Mock oracle not found at {_MOCK_ORACLE}."
    )


@dataclass
class DatasetConfig:
    """Signal availability for a dataset."""

    name: str
    object_name: str  # e.g., "videos", "books", "news articles"
    has_explicit_positive: bool = True
    has_implicit_positive: bool = False
    has_implicit_negative: bool = False
    has_explicit_negative: bool = False


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "synthetic": DatasetConfig(
        name="synthetic",
        object_name="videos",
        has_explicit_positive=True,
        has_implicit_positive=True,
        has_implicit_negative=True,
    ),
    "kuairec": DatasetConfig(
        name="kuairec",
        object_name="videos",
        has_explicit_positive=True,
        has_implicit_positive=True,
        has_implicit_negative=True,
    ),
    "mind": DatasetConfig(
        name="mind",
        object_name="news articles",
        has_explicit_positive=True,
        has_implicit_negative=True,
    ),
    "amazon_digital_music": DatasetConfig(
        name="amazon_digital_music",
        object_name="songs",
        has_explicit_positive=True,
        has_implicit_positive=True,
        has_explicit_negative=True,
    ),
    "yelp": DatasetConfig(
        name="yelp",
        object_name="stores",
        has_explicit_positive=True,
        has_implicit_positive=True,
        has_explicit_negative=True,
    ),
    "goodreads": DatasetConfig(
        name="goodreads",
        object_name="books",
        has_explicit_positive=True,
        has_implicit_positive=True,
        has_explicit_negative=True,
    ),
}
