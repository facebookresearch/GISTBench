"""Interest Specificity (IS) evaluation.

Implements the IS evaluation from the paper:
1. Stage 1: Build global pool, LLM removes semantically overlapping objects
2. Stage 2: For each interest, build test set (backing + distractors from shortlisted pool)
3. LLM judge identifies which objects support the interest
4. Compute correct (judge got right) and backing (total evidence count)

Reference: paper Section "Interest Specificity"
"""

from __future__ import annotations

import logging
import random
import re

import pandas as pd

from gistbench.client import LLMClient
from gistbench.schema import DatasetConfig, ISResult, Interest

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

IS_STAGE1_SYSTEM_PROMPT = (
    "You are an expert in semantic similarity and interest matching.\n\n"
    "Given a user's interests and a pool of objects with associated interests, "
    "identify which objects should be REMOVED due to overlapping interests.\n\n"
    'Consider direct matches and semantic similarity (e.g., "football" ≈ "soccer").\n\n'
    'Output: Comma-separated object indices to REMOVE, or "NONE" if no overlaps.'
)

IS_JUDGE_SYSTEM_PROMPT = (
    "You are an expert in content analysis and user interest understanding.\n\n"
    "Given an interest and item descriptions, identify the items "
    "that most likely led to inferring this interest.\n\n"
    'Output ONLY item IDs (e.g., "item_0, item_12"), comma-separated. '
    "No explanation."
)

# Default test set size (backing + distractors)
DEFAULT_TEST_SET_SIZE = 50
DEFAULT_MAX_BACKING = 5
DEFAULT_POOL_SIZE = 1000


def shortlist_pool(
    user_interests: list[str],
    pool_df: pd.DataFrame,
    client: LLMClient | None,
    pool_size: int = DEFAULT_POOL_SIZE,
    seed: int = 42,
) -> pd.DataFrame:
    """IS Stage 1: Build global pool and remove semantically overlapping objects.

    Args:
        user_interests: List of interest names for this user.
        pool_df: DataFrame of candidate distractor objects (must have object_id, object_text).
        client: LLM client for shortlisting. If None, skips shortlisting.
        pool_size: Max objects to sample for the pool.
        seed: Random seed for deterministic sampling.

    Returns:
        Shortlisted pool DataFrame with overlapping objects removed.
    """
    # Sample pool deterministically
    pool_unique = pool_df.drop_duplicates(subset=["object_id"])
    if len(pool_unique) > pool_size:
        pool_unique = pool_unique.sample(n=pool_size, random_state=seed)

    if client is None or not user_interests:
        return pool_unique

    # Build pool description: each object described by its text (truncated)
    pool_items = []
    idx_to_oid = {}
    for idx, (_, row) in enumerate(pool_unique.iterrows()):
        text = str(row.get("object_text", ""))[:200]
        pool_items.append(f"Object {idx}: {text}")
        idx_to_oid[idx] = row["object_id"]

    user_prompt = (
        "User's interests: " + ", ".join(user_interests)
        + "\n\nGlobal pool of items:\n" + "\n".join(pool_items)
        + '\n\nWhich object indices are too similar? Output comma-separated indices or "NONE".'
    )

    messages = [
        {"role": "system", "content": IS_STAGE1_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    retry_messages = list(messages)
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat(retry_messages)
            response_upper = response.strip().upper()

            if "NONE" in response_upper:
                return pool_unique

            remove_indices = re.findall(r"\d+", response)
            remove_oids = set()
            for idx_str in remove_indices:
                idx = int(idx_str)
                if idx in idx_to_oid:
                    remove_oids.add(idx_to_oid[idx])

            if remove_oids:
                shortlisted = pool_unique[~pool_unique["object_id"].isin(remove_oids)]
                logger.info(
                    "IS Stage 1: removed %d/%d objects from pool",
                    len(remove_oids), len(pool_unique),
                )
                return shortlisted

            logger.warning(
                "IS Stage 1: no parseable indices in response (attempt %d/%d)",
                attempt + 1, MAX_RETRIES,
            )
            retry_messages = messages + [
                {"role": "user", "content": (
                    "Error: could not parse your response. "
                    'Output ONLY comma-separated object indices to remove, or "NONE".'
                )},
            ]

        except Exception as e:
            last_err = e
            logger.warning(
                "IS Stage 1 shortlisting call failed (attempt %d/%d)",
                attempt + 1, MAX_RETRIES,
            )

    if last_err is not None:
        raise RuntimeError(
            f"IS Stage 1 shortlisting failed after {MAX_RETRIES} retries"
        ) from last_err
    return pool_unique


def _build_test_set(
    interest: Interest,
    user_df: pd.DataFrame,
    shortlisted_pool: pd.DataFrame,
    test_set_size: int = DEFAULT_TEST_SET_SIZE,
    max_backing: int = DEFAULT_MAX_BACKING,
    seed: int = 42,
) -> tuple[list[dict], list[str]]:
    """Build a shuffled test set of backing + distractor objects.

    Returns:
        (test_set_items, backing_ids)
        - test_set_items: list of dicts with 'object_id', 'object_text', 'is_backing'
        - backing_ids: list of object_ids that are backing (evidence) items
    """
    rng = random.Random(seed)

    # Get backing objects (cited by the interest)
    backing_ids = set(interest.item_ids)
    backing_df = user_df[user_df["object_id"].isin(backing_ids)].drop_duplicates(
        subset=["object_id"]
    )

    # Sample backing if too many
    if len(backing_df) > max_backing:
        backing_df = backing_df.sample(n=max_backing, random_state=seed)

    actual_backing_ids = set(backing_df["object_id"].values)
    num_backing = len(actual_backing_ids)

    # Get distractor objects from the shortlisted pool (not in backing)
    num_distractors = test_set_size - num_backing
    distractor_candidates = shortlisted_pool[
        ~shortlisted_pool["object_id"].isin(actual_backing_ids)
    ].drop_duplicates(subset=["object_id"])

    if len(distractor_candidates) < num_distractors:
        num_distractors = len(distractor_candidates)

    if num_distractors > 0:
        distractor_df = distractor_candidates.sample(
            n=num_distractors, random_state=seed
        )
    else:
        distractor_df = distractor_candidates.iloc[:0]

    # Build test set
    test_items = []
    for _, row in backing_df.iterrows():
        test_items.append({
            "object_id": row["object_id"],
            "object_text": row.get("object_text", ""),
            "is_backing": True,
        })
    for _, row in distractor_df.iterrows():
        test_items.append({
            "object_id": row["object_id"],
            "object_text": row.get("object_text", ""),
            "is_backing": False,
        })

    # Shuffle
    rng.shuffle(test_items)

    return test_items, list(actual_backing_ids)


def _judge_identify_backing(
    interest: Interest,
    test_items: list[dict],
    num_backing: int,
    client: LLMClient,
) -> list[str]:
    """Ask LLM judge to identify which items in the test set support the interest.

    Returns list of object_ids selected by the judge.
    """
    # Build prompt
    item_lines = []
    idx_to_id = {}
    for idx, item in enumerate(test_items):
        item_lines.append(f"item_{idx}: {item['object_text']}")
        idx_to_id[idx] = item["object_id"]

    user_prompt = (
        f'Interest: "{interest.name}"\n\n'
        f"Items:\n{chr(10).join(item_lines)}\n\n"
        f"Which {num_backing} items support this interest?"
    )

    messages = [
        {"role": "system", "content": IS_JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    retry_messages = list(messages)
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat(retry_messages)

            selected_indices = re.findall(r"item[_\s]*(\d+)", response.lower())
            selected_ids = []
            for idx_str in selected_indices:
                idx = int(idx_str)
                if idx in idx_to_id:
                    selected_ids.append(idx_to_id[idx])

            if selected_ids:
                return list(set(selected_ids))

            logger.warning(
                "IS judge returned no parseable item references for interest '%s' (attempt %d/%d)",
                interest.name, attempt + 1, MAX_RETRIES,
            )
            retry_messages = messages + [
                {"role": "user", "content": (
                    "Error: could not parse your response. "
                    'Output ONLY item IDs (e.g., "item_0, item_12"), comma-separated.'
                )},
            ]

        except Exception as e:
            last_err = e
            logger.warning(
                "IS judge call failed for interest '%s' (attempt %d/%d)",
                interest.name, attempt + 1, MAX_RETRIES,
            )

    if last_err is not None:
        raise RuntimeError(
            f"IS judge failed for interest '{interest.name}' after {MAX_RETRIES} retries"
        ) from last_err
    return []


def evaluate_interest_specificity(
    interest: Interest,
    user_df: pd.DataFrame,
    shortlisted_pool: pd.DataFrame,
    client: LLMClient,
    test_set_size: int = DEFAULT_TEST_SET_SIZE,
    max_backing: int = DEFAULT_MAX_BACKING,
    seed: int = 42,
) -> tuple[int, int, int]:
    """Evaluate specificity for a single interest using LLM judge retrieval.

    Returns:
        (correct, selected, backing)
        - correct: number of judge selections that are actually backing objects
        - selected: total number of objects the judge selected
        - backing: total number of backing objects in the test set
    """
    test_items, backing_ids = _build_test_set(
        interest, user_df, shortlisted_pool,
        test_set_size=test_set_size,
        max_backing=max_backing,
        seed=seed,
    )

    if not test_items or not backing_ids:
        return 0, 0, 0

    num_backing = len(backing_ids)

    # LLM judge identifies backing items
    selected_ids = _judge_identify_backing(
        interest, test_items, num_backing, client
    )

    # Count correct
    backing_set = set(backing_ids)
    correct = len(set(selected_ids) & backing_set)
    selected = len(selected_ids)

    return correct, selected, num_backing


def evaluate_is(
    interests: list[Interest],
    user_id: str,
    user_df: pd.DataFrame,
    config: DatasetConfig,
    client: LLMClient,
    model_name: str = "",
    pool_df: pd.DataFrame | None = None,
    test_set_size: int = DEFAULT_TEST_SET_SIZE,
    max_backing: int = DEFAULT_MAX_BACKING,
    pool_size: int = DEFAULT_POOL_SIZE,
    seed: int = 42,
) -> list[ISResult]:
    """Evaluate Interest Specificity for all predicted interests.

    Uses IS Stage 1 (shortlisting) + Stage 2 (identification):
    1. Build global pool, LLM removes semantically overlapping objects
    2. For each interest, build test set with backing + distractors from shortlisted pool
    3. LLM judge identifies backing objects

    Args:
        interests: Predicted interests with cited item_ids.
        user_id: User identifier.
        user_df: DataFrame of user's engagement history.
        config: Dataset configuration.
        client: LLM client for IS judge evaluation.
        model_name: Name of the model being evaluated.
        pool_df: Global pool of objects for distractors. If None, uses user_df.
        test_set_size: Total items in test set (backing + distractors).
        max_backing: Max backing items sampled per interest.
        pool_size: Max objects to sample for Stage 1 pool.
        seed: Random seed.
    """
    # Use provided pool or fall back to user's own objects
    if pool_df is None:
        pool_df = user_df

    # Stage 1: Shortlist the pool
    user_interest_names = [i.name for i in interests]
    shortlisted = shortlist_pool(
        user_interest_names, pool_df, client,
        pool_size=pool_size, seed=seed,
    )

    # Stage 2: Evaluate each interest (per-interest seed for independent distractors)
    results = []
    for idx, interest in enumerate(interests):
        interest_seed = hash((seed, interest.name, idx)) % (2**31)
        correct, selected, backing = evaluate_interest_specificity(
            interest, user_df, shortlisted, client,
            test_set_size=test_set_size,
            max_backing=max_backing,
            seed=interest_seed,
        )
        results.append(
            ISResult(
                user_id=user_id,
                dataset=config.name,
                model=model_name,
                interest=interest.name,
                correct=correct,
                selected=selected,
                backing=backing,
            )
        )
    return results
