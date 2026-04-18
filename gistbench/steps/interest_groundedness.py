"""Interest Groundedness (IG) verification.

Implements the IG evaluation from the paper:
1. LLM judge filters cited evidence for semantic relevance
2. Dataset-specific verification predicates check engagement counts
3. Produces a binary verified flag per interest

Reference: paper Section "Interest Groundedness"
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from gistbench.client import LLMClient
from gistbench.schema import DatasetConfig, IGResult, Interest

logger = logging.getLogger(__name__)

MAX_RETRIES = 3

IG_JUDGE_SYSTEM_PROMPT = (
    "You are an expert in evaluating whether evidence supports inferred interests.\n\n"
    "Given a user's interest and a list of items they engaged with, determine which items "
    "are TRULY RELEVANT to the interest (not just tangentially related).\n\n"
    "For each item, consider:\n"
    "- Does the item content directly relate to the interest?\n"
    "- Would engaging with this item plausibly lead to inferring this interest?\n\n"
    'Output ONLY comma-separated indices of relevant items (e.g., "0, 2, 5"), '
    'or "NONE" if no items are relevant. NO explanation. '
    "ONLY comma-separated indices of relevant items, nothing else."
)

IG_FEW_SHOT_EXAMPLES = [
    {
        "user": (
            'Interest: "Fitness & Gym Culture"\n\n'
            "Items the user engaged with:\n"
            "Item 0: #workout #gym #fitness #bodybuilding #gains\n"
            "Item 1: #cooking #recipes #foodie #homemade\n"
            "Item 2: #crossfit #wod #fitfam #gymlife\n"
            "Item 3: #travel #vacation #beach #summer\n\n"
            "Which items are RELEVANT to this interest? "
            'Output comma-separated indices (e.g., "0, 2, 5") or "NONE" if none are relevant.'
        ),
        "assistant": "0, 2",
    },
    {
        "user": (
            'Interest: "Classical Music Appreciation"\n\n'
            "Items the user engaged with:\n"
            "Item 0: #hiphop #rap #beats #newmusic\n"
            "Item 1: #gaming #esports #twitch #streamer\n"
            "Item 2: #football #nfl #touchdown #sports\n\n"
            "Which items are RELEVANT to this interest? "
            'Output comma-separated indices (e.g., "0, 2, 5") or "NONE" if none are relevant.'
        ),
        "assistant": "NONE",
    },
]


def _filter_citations_with_judge(
    interest: Interest,
    user_df: pd.DataFrame,
    client: LLMClient | None,
) -> list[str]:
    """Filter cited evidence objects using LLM judge for semantic relevance.

    Returns the subset of item_ids deemed relevant by the judge.
    If no client is provided, returns all item_ids (skip filtering).
    """
    if client is None:
        return interest.item_ids

    cited_df = user_df[user_df["object_id"].isin(interest.item_ids)]
    if cited_df.empty:
        return []

    # Build prompt
    items_text = []
    idx_to_id = {}
    for idx, (_, row) in enumerate(cited_df.iterrows()):
        items_text.append(f"Item {idx}: {row.get('object_text', '')}")
        idx_to_id[idx] = row["object_id"]

    user_prompt = (
        f'Interest: "{interest.name}"\n\n'
        f"Items:\n{chr(10).join(items_text)}\n\n"
        f"Which items are RELEVANT? Output comma-separated indices or \"NONE\"."
    )

    messages = [
        {"role": "system", "content": IG_JUDGE_SYSTEM_PROMPT},
    ]
    # Add few-shot examples
    for example in IG_FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": example["user"]})
        messages.append({"role": "assistant", "content": example["assistant"]})
    messages.append({"role": "user", "content": user_prompt})

    retry_messages = list(messages)
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat(retry_messages)
            response_upper = response.strip().upper()

            if "NONE" in response_upper:
                return []

            indices = re.findall(r"\d+", response)
            relevant_ids = []
            for idx_str in indices:
                idx = int(idx_str)
                if idx in idx_to_id:
                    relevant_ids.append(idx_to_id[idx])

            if relevant_ids:
                return relevant_ids

            logger.warning(
                "IG judge returned no parseable indices for interest '%s' (attempt %d/%d)",
                interest.name, attempt + 1, MAX_RETRIES,
            )
            retry_messages = messages + [
                {"role": "user", "content": (
                    "Error: could not parse your response. "
                    'Output ONLY comma-separated indices (e.g., "0, 2, 5") or "NONE".'
                )},
            ]

        except Exception as e:
            last_err = e
            logger.warning(
                "IG judge call failed for interest '%s' (attempt %d/%d)",
                interest.name, attempt + 1, MAX_RETRIES,
            )

    if last_err is not None:
        raise RuntimeError(
            f"IG judge failed for interest '{interest.name}' after {MAX_RETRIES} retries"
        ) from last_err
    return []


def _count_by_type(
    df: pd.DataFrame, item_ids: list[str], interaction_type: str
) -> int:
    """Count unique objects of a given interaction type among the specified item IDs."""
    mask = (df["object_id"].isin(item_ids)) & (
        df["interaction_type"] == interaction_type
    )
    return int(df.loc[mask, "object_id"].nunique())


def verify_interest(
    interest: Interest,
    filtered_ids: list[str],
    user_df: pd.DataFrame,
    config: DatasetConfig,
) -> bool:
    """Check if an interest meets the dataset-specific verification predicate.

    Uses the judge-filtered evidence set (filtered_ids) for counting.

    Positive evidence (at least one must hold):
      - >= 3 implicit_positive engagements (if dataset has them)
      - >= 2 explicit_positive engagements (if dataset has them)
      - >= 1 explicit_positive + >= 2 implicit_positive (if both exist)

    Negative constraints (all must hold):
      - <= 3 implicit_negative engagements (if dataset has them)
      - <= 2 explicit_negative engagements (if dataset has them)
    """
    if not filtered_ids:
        return False

    n_explicit_pos = _count_by_type(user_df, filtered_ids, "explicit_positive")
    n_implicit_pos = _count_by_type(user_df, filtered_ids, "implicit_positive")
    n_explicit_neg = _count_by_type(user_df, filtered_ids, "explicit_negative")
    n_implicit_neg = _count_by_type(user_df, filtered_ids, "implicit_negative")

    # Negative constraints
    if config.has_implicit_negative and n_implicit_neg > 3:
        return False
    if config.has_explicit_negative and n_explicit_neg > 2:
        return False

    # Positive evidence — at least one condition must be satisfied
    conditions: list[bool] = []

    if config.has_implicit_positive:
        conditions.append(n_implicit_pos >= 3)
    if config.has_explicit_positive:
        conditions.append(n_explicit_pos >= 2)
    if config.has_implicit_positive and config.has_explicit_positive:
        conditions.append(n_explicit_pos >= 1 and n_implicit_pos >= 2)

    return any(conditions)


def evaluate_ig(
    interests: list[Interest],
    user_id: str,
    user_df: pd.DataFrame,
    config: DatasetConfig,
    model_name: str = "",
    judge_client: LLMClient | None = None,
) -> list[IGResult]:
    """Evaluate Interest Groundedness for all predicted interests.

    Args:
        interests: Predicted interests with cited item_ids.
        user_id: User identifier.
        user_df: DataFrame of user's engagement history.
        config: Dataset configuration.
        model_name: Name of the model being evaluated.
        judge_client: Optional LLM client for IG evidence filtering.
            If None, skips LLM judge filtering (uses all cited items).
    """
    results = []
    for interest in interests:
        # Step 1: LLM judge filters citations for relevance
        filtered_ids = _filter_citations_with_judge(
            interest, user_df, judge_client
        )

        # Step 2: Apply verification predicate
        verified = verify_interest(interest, filtered_ids, user_df, config)
        results.append(
            IGResult(
                user_id=user_id,
                dataset=config.name,
                model=model_name,
                interest=interest.name,
                verified=verified,
            )
        )
    return results
