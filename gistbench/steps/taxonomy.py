"""Interest Category Mapping.

Maps fine-grained predicted interests to broader interest categories using
an LLM.  This normalization prevents score inflation from producing many
interests within the same category — only breadth across categories matters.

Reference: paper Section "Interest Category Normalization"

The bundled ``categories.csv`` contains 325 interest categories, each with
a stable ``category_id``.  All normalization and oracle computation uses
category IDs, not names — IDs are deterministic and avoid string-matching
issues.
"""

from __future__ import annotations

import csv
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from gistbench.client import LLMClient

logger = logging.getLogger(__name__)

# Path to the bundled taxonomy (325 interest categories)
_BUNDLED_TAXONOMY = Path(__file__).resolve().parent.parent / "assets" / "categories.csv"


@dataclass
class Taxonomy:
    """Bidirectional mapping between category IDs and names.

    The canonical representation — all scoring, oracle, and caching
    operations use ``category_id`` (int) as the key.
    """

    id_to_name: dict[int, str] = field(default_factory=dict)
    name_to_id: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.id_to_name)

    def names(self) -> list[str]:
        """All category names in ID order."""
        return [self.id_to_name[i] for i in sorted(self.id_to_name)]

    def ids(self) -> list[int]:
        """All category IDs in sorted order."""
        return sorted(self.id_to_name)

    @classmethod
    def from_csv(cls, path: str | Path) -> "Taxonomy":
        """Load from a CSV with ``category_id,category_name`` columns."""
        path = Path(path)
        id_to_name: dict[int, str] = {}
        name_to_id: dict[str, int] = {}
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            if "category_id" not in (reader.fieldnames or []):
                raise ValueError(
                    f"CSV must have 'category_id' and 'category_name' columns. "
                    f"Found: {reader.fieldnames}"
                )
            for row in reader:
                cid = int(row["category_id"])
                cname = row["category_name"]
                id_to_name[cid] = cname
                name_to_id[cname] = cid
        logger.info("Loaded taxonomy with %d categories from %s", len(id_to_name), path)
        return cls(id_to_name=id_to_name, name_to_id=name_to_id)

    @classmethod
    def from_list(cls, names: list[str]) -> "Taxonomy":
        """Create from a plain list (IDs assigned 1..N in order)."""
        id_to_name = {i + 1: n for i, n in enumerate(names)}
        name_to_id = {n: i + 1 for i, n in enumerate(names)}
        return cls(id_to_name=id_to_name, name_to_id=name_to_id)


def load_default_taxonomy() -> Taxonomy:
    """Load the bundled 325-category taxonomy shipped with GISTBench.

    Returns:
        Taxonomy with stable category IDs.

    Raises:
        FileNotFoundError: If ``categories.csv`` is missing from assets.
    """
    if not _BUNDLED_TAXONOMY.exists():
        raise FileNotFoundError(
            f"Bundled taxonomy not found at {_BUNDLED_TAXONOMY}. "
            "Ensure gistbench/assets/categories.csv is present."
        )
    return Taxonomy.from_csv(_BUNDLED_TAXONOMY)


TAXONOMY_SYSTEM_PROMPT = (
    "You are an expert at categorizing user interests into a standardized taxonomy.\n\n"
    "## Task\n"
    "You are mapping SPECIFIC predicted interests to BROADER interest category names. "
    "The predicted interests are fine-grained (e.g., \"LeBron James\", "
    "\"Attack on Titan\", \"ChatGPT tips\") and must be mapped to their parent "
    "interest categories (e.g., \"Basketball\", \"Anime\", \"Artificial Intelligence\"). "
    "Map each predicted interest to ONE category from the NUMBERED list provided below.\n\n"
    "## CRITICAL RULES:\n"
    "1. You MUST ONLY use category numbers from the list below\n"
    "2. Output ONLY the interest INDEX and category NUMBER - nothing else\n"
    "3. DO NOT output category names or interest text - only output NUMBERS\n"
    "4. If unsure, pick the CLOSEST match from the list\n"
    "5. EVERY interest MUST be mapped to a category - no exceptions\n\n"
    "## Available Interest Categories (use the NUMBER, not the name):\n"
    "{categories}\n\n"
    "## Output Format:\n"
    "<interest_index>: <category_number>\n\n"
    "One mapping per line. No explanations."
)

BATCH_SIZE = 3
MAX_RETRIES = 3

# Sentinel category_id for unmapped interests
OTHER_CATEGORY_ID = 0


def _build_category_list(taxonomy: Taxonomy) -> str:
    """Build a numbered category list string for the LLM prompt."""
    return "\n".join(
        f"{cid}. {taxonomy.id_to_name[cid]}"
        for cid in sorted(taxonomy.id_to_name)
    )


def _parse_taxonomy_response(
    response: str,
    interest_names: list[str],
    taxonomy: Taxonomy,
) -> dict[str, int]:
    """Parse LLM taxonomy response into interest -> category_id mapping.

    Expected format: one line per interest, each line is "<interest_index>: <category_id>"
    """
    mapping: dict[str, int] = {}
    for line in response.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"(\d+)\s*:\s*(\d+)", line)
        if match:
            interest_idx = int(match.group(1)) - 1  # 1-indexed in prompt
            cat_id = int(match.group(2))
            if 0 <= interest_idx < len(interest_names) and cat_id in taxonomy.id_to_name:
                mapping[interest_names[interest_idx]] = cat_id
    return mapping


def map_interests_to_categories(
    interest_names: list[str],
    client: LLMClient,
    taxonomy: Taxonomy,
    batch_size: int = BATCH_SIZE,
) -> dict[str, int]:
    """Map fine-grained interest names to category IDs using an LLM.

    Args:
        interest_names: List of unique interest names to map.
        client: LLM client for taxonomy mapping.
        taxonomy: Taxonomy with category IDs and names.
        batch_size: Number of interests per LLM call.

    Returns:
        Dict mapping interest name -> category_id.
        Unmapped interests get ``OTHER_CATEGORY_ID`` (0).
    """
    if not interest_names:
        return {}

    if len(taxonomy) == 0:
        raise ValueError(
            "Taxonomy is empty. Use load_default_taxonomy() or "
            "Taxonomy.from_csv() to load categories."
        )

    cat_list_str = _build_category_list(taxonomy)
    system_prompt = TAXONOMY_SYSTEM_PROMPT.format(categories=cat_list_str)

    mapping: dict[str, int] = {}

    for batch_start in range(0, len(interest_names), batch_size):
        batch = interest_names[batch_start : batch_start + batch_size]

        # Build user prompt with 1-indexed interests
        interest_lines = [f"{i + 1}. {name}" for i, name in enumerate(batch)]
        user_prompt = "Map these interests:\n" + "\n".join(interest_lines)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_err: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat(messages)
                batch_mapping = _parse_taxonomy_response(response, batch, taxonomy)
                mapping.update(batch_mapping)
                break
            except Exception as e:
                last_err = e
                logger.warning(
                    "Taxonomy mapping call failed for batch at %d (attempt %d/%d)",
                    batch_start, attempt + 1, MAX_RETRIES,
                )
        else:
            raise RuntimeError(
                f"Taxonomy mapping failed for batch at {batch_start} after {MAX_RETRIES} retries"
            ) from last_err

    # Fill unmapped interests with OTHER
    for name in interest_names:
        if name not in mapping:
            mapping[name] = OTHER_CATEGORY_ID

    return mapping


class TaxonomyCache:
    """SQLite-backed cache for interest → category_id mappings.

    Avoids redundant LLM calls by persisting previously mapped interests.
    The cache stores ``category_id`` (int), not names.

    Usage::

        taxonomy = load_default_taxonomy()
        cache = TaxonomyCache("taxonomy_cache.db")
        mapping = cache.map(interest_names, client, taxonomy)
    """

    def __init__(self, db_path: str | Path = "taxonomy_cache.db") -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS taxonomy_map ("
            "  interest TEXT PRIMARY KEY,"
            "  category_id INTEGER NOT NULL"
            ")"
        )
        self._conn.commit()

    def get(self, interest: str) -> int | None:
        """Look up a cached mapping. Returns None if not cached."""
        row = self._conn.execute(
            "SELECT category_id FROM taxonomy_map WHERE interest = ?",
            (interest,),
        ).fetchone()
        return row[0] if row else None

    def get_many(self, interests: list[str]) -> dict[str, int]:
        """Look up multiple interests. Returns dict of cached mappings."""
        if not interests:
            return {}
        placeholders = ",".join("?" * len(interests))
        rows = self._conn.execute(
            f"SELECT interest, category_id FROM taxonomy_map WHERE interest IN ({placeholders})",
            interests,
        ).fetchall()
        return dict(rows)

    def put(self, interest: str, category_id: int) -> None:
        """Cache a single mapping (upsert)."""
        self._conn.execute(
            "INSERT OR REPLACE INTO taxonomy_map (interest, category_id) VALUES (?, ?)",
            (interest, category_id),
        )
        self._conn.commit()

    def put_many(self, mapping: dict[str, int]) -> None:
        """Cache multiple mappings (upsert)."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO taxonomy_map (interest, category_id) VALUES (?, ?)",
            list(mapping.items()),
        )
        self._conn.commit()

    def all_mappings(self) -> dict[str, int]:
        """Return all cached mappings."""
        rows = self._conn.execute(
            "SELECT interest, category_id FROM taxonomy_map ORDER BY interest"
        ).fetchall()
        return dict(rows)

    def count(self) -> int:
        """Number of cached mappings."""
        row = self._conn.execute("SELECT COUNT(*) FROM taxonomy_map").fetchone()
        return row[0]

    def map(
        self,
        interest_names: list[str],
        client: LLMClient,
        taxonomy: Taxonomy,
        batch_size: int = BATCH_SIZE,
    ) -> dict[str, int]:
        """Map interests to category IDs, using cache for known interests.

        Only calls the LLM for interests not already in the cache.
        New mappings are automatically persisted to the database.

        Returns:
            Dict mapping interest name -> category_id.
        """
        if not interest_names:
            return {}

        cached = self.get_many(interest_names)
        uncached = [n for n in interest_names if n not in cached]

        if uncached:
            logger.info(
                "Taxonomy cache: %d cached, %d to map via LLM",
                len(cached), len(uncached),
            )
            new_mappings = map_interests_to_categories(
                uncached, client, taxonomy, batch_size=batch_size,
            )
            self.put_many(new_mappings)
            cached.update(new_mappings)
        else:
            logger.info("Taxonomy cache: all %d interests found in cache", len(cached))

        return {n: cached.get(n, OTHER_CATEGORY_ID) for n in interest_names}

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "TaxonomyCache":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
