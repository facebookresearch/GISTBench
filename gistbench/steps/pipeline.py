"""End-to-end GISTBench evaluation pipeline.

Implements the full pipeline from the paper:
  Branch A: Interest extraction → IG judge filtering → IG verification
                                → IS shortlisting → IS identification
  Branch B: Interest category mapping (interests → taxonomy categories)
  Merge:    Score aggregation (IG_P, IG_R, IG_F1, IS_Verified)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from gistbench.store import ResultsStore

from gistbench.client import LLMClient, parse_json_response
from gistbench.data import chunk_user_history, detect_dataset_config
from gistbench.prompts.interest_extraction import build_extraction_messages
from gistbench.schema import DatasetConfig, IGResult, ISResult, Interest, Oracle, UserScore
from gistbench.steps.interest_groundedness import evaluate_ig
from gistbench.steps.interest_specificity import evaluate_is
from gistbench.steps.scoring import compute_user_score
from gistbench.steps.taxonomy import Taxonomy, TaxonomyCache, load_default_taxonomy, map_interests_to_categories

logger = logging.getLogger(__name__)


@dataclass
class UserEvalResult:
    """Full evaluation output for a single user, including intermediate results."""

    score: UserScore
    ig_results: list[IGResult] = field(default_factory=list)
    is_results: list[ISResult] = field(default_factory=list)
    category_map: dict[str, int] | None = None


def parse_interests(raw: list | dict | None) -> list[Interest]:
    """Convert raw JSON from LLM into Interest objects."""
    if not raw or not isinstance(raw, list):
        return []

    interests = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        interests.append(
            Interest(
                name=item.get("interest", ""),
                item_ids=item.get("item_ids", []),
                evidence_excerpt=item.get("evidence_excerpt", ""),
                brief_rationale=item.get("brief_rationale", ""),
            )
        )
    return interests


def evaluate_user(
    client: LLMClient,
    df: pd.DataFrame,
    user_id: str,
    config: DatasetConfig,
    oracle: Oracle | None = None,
    model_name: str = "",
    chunk_size: int = 100,
    judge_client: LLMClient | None = None,
    use_judge: bool = True,
    use_taxonomy: bool = True,
    taxonomy_client: LLMClient | None = None,
    taxonomy: Taxonomy | None = None,
    taxonomy_cache: TaxonomyCache | None = None,
    pool_df: pd.DataFrame | None = None,
    test_set_size: int = 50,
    max_backing: int = 5,
    pool_size: int = 1000,
) -> UserEvalResult:
    """Run the full pipeline for a single user.

    Args:
        client: LLM client for interest extraction.
        df: Full dataset DataFrame.
        user_id: User to evaluate.
        config: Dataset configuration.
        oracle: Oracle with per-user ground-truth category IDs.
            Provides the denominator for IG Recall.  If None, extraction
            and IG/IS evaluation still run but scoring is skipped (scores
            are all zero).  Use this for the first pass
            where the oracle is computed *after* all models have been evaluated.
        model_name: Name of the model being evaluated.
        chunk_size: Max engagements per extraction prompt.
        judge_client: LLM client for IG/IS judge evaluation.
            If None and use_judge=True, uses the same client.
        use_judge: Whether to use LLM judge for IG filtering and IS retrieval.
        use_taxonomy: Whether to use interest category mapping for score normalization.
        taxonomy_client: LLM client for taxonomy mapping. If None, uses judge_client or client.
        taxonomy: Taxonomy with category IDs. If None, uses bundled taxonomy.
        taxonomy_cache: Optional SQLite-backed cache for taxonomy mappings.
            If provided, cached mappings are reused and new ones are persisted.
        pool_df: Global pool of objects for IS distractors. If None, uses user's objects.
        test_set_size: IS test set size (backing + distractors).
        max_backing: Max backing items sampled per interest for IS.
        pool_size: Max objects sampled for IS distractor pool.

    Returns:
        UserEvalResult with the score plus intermediate IG/IS results and category map.
    """
    chunks = chunk_user_history(df, user_id, chunk_size=chunk_size)
    user_df = df[df["user_id"] == user_id]

    raw_interests: list[Interest] = []

    for chunk in chunks:
        engagements = chunk.to_dict("records")
        messages = build_extraction_messages(config, engagements)

        response = client.chat(messages)
        parsed = parse_json_response(response)
        if parsed is None:
            raise ValueError(
                f"Failed to parse JSON from LLM response for user {user_id} "
                f"(chunk {chunks.index(chunk) + 1}/{len(chunks)}). "
                f"Raw response: {response[:200]}"
            )
        interests = parse_interests(parsed)
        raw_interests.extend(interests)

    # Deduplicate interests by name, merging item_ids from multiple chunks
    merged: dict[str, Interest] = {}
    for interest in raw_interests:
        if interest.name in merged:
            existing = merged[interest.name]
            seen_ids = set(existing.item_ids)
            for iid in interest.item_ids:
                if iid not in seen_ids:
                    existing.item_ids.append(iid)
                    seen_ids.add(iid)
        else:
            merged[interest.name] = interest
    all_interests = list(merged.values())

    logger.info(
        "User %s: extracted %d interests (%d before dedup) from %d chunks",
        user_id,
        len(all_interests),
        len(raw_interests),
        len(chunks),
    )

    # Determine judge client
    effective_judge = judge_client if judge_client else (client if use_judge else None)

    # Stage 1a: IG — judge filtering + verification
    ig_results = evaluate_ig(
        all_interests, user_id, user_df, config, model_name,
        judge_client=effective_judge,
    )

    # Stage 1b: IS — shortlisting + retrieval-based evaluation
    if use_judge and effective_judge:
        is_results = evaluate_is(
            all_interests, user_id, user_df, config, effective_judge,
            model_name=model_name,
            pool_df=pool_df,
            test_set_size=test_set_size,
            max_backing=max_backing,
            pool_size=pool_size,
        )
    else:
        # No judge: IS cannot be meaningfully evaluated, record zeros
        is_results = [
            ISResult(
                user_id=user_id,
                dataset=config.name,
                model=model_name,
                interest=interest.name,
                correct=0,
                selected=0,
                backing=0,
            )
            for interest in all_interests
        ]

    # Branch B: Interest category mapping
    category_map = None
    if use_taxonomy and all_interests:
        tax = taxonomy if taxonomy else load_default_taxonomy()
        tax_client = taxonomy_client or effective_judge or client
        unique_names = list({i.name for i in all_interests})
        if taxonomy_cache is not None:
            category_map = taxonomy_cache.map(
                unique_names, tax_client, taxonomy=tax,
            )
        else:
            category_map = map_interests_to_categories(
                unique_names, tax_client, taxonomy=tax,
            )
        logger.info(
            "User %s: mapped %d interests to %d categories",
            user_id,
            len(unique_names),
            len(set(category_map.values())),
        )

    # Score — requires oracle
    if oracle is not None:
        oracle_count = oracle.count(user_id)
        if oracle_count == 0:
            raise ValueError(
                f"User '{user_id}' has no oracle entry. "
                f"Provide an oracle that covers this user."
            )
        score = compute_user_score(
            ig_results, is_results, user_id, config.name, model_name,
            oracle_count=oracle_count,
            category_map=category_map,
        )
    else:
        # No oracle — return placeholder score; real scores come from rescore_all()
        score = UserScore(user_id=user_id, dataset=config.name, model=model_name)

    return UserEvalResult(
        score=score,
        ig_results=ig_results,
        is_results=is_results,
        category_map=category_map,
    )


def run_benchmark(
    client: LLMClient,
    df: pd.DataFrame,
    user_ids: list[str],
    oracle: Oracle | None = None,
    dataset_name: str = "",
    model_name: str = "",
    chunk_size: int = 100,
    judge_client: LLMClient | None = None,
    use_judge: bool = True,
    use_taxonomy: bool = True,
    taxonomy_client: LLMClient | None = None,
    taxonomy: Taxonomy | None = None,
    taxonomy_cache: TaxonomyCache | None = None,
    results_store: "ResultsStore | None" = None,
    test_set_size: int = 50,
    max_backing: int = 5,
    pool_size: int = 1000,
) -> list[UserScore]:
    """Run GISTBench evaluation for a list of users.

    Args:
        client: LLM client for interest extraction.
        df: Full dataset DataFrame.
        user_ids: List of user IDs to evaluate.
        oracle: Oracle with per-user ground-truth interest categories.
            Provides the denominator for IG Recall.  If None, extraction
            and IG/IS evaluation still run but scoring is deferred — use
            ``results_store`` to persist results and call
            ``results_store.rescore_all()`` after building the oracle.
        dataset_name: Dataset name for config lookup.
        model_name: Name of the model being evaluated.
        chunk_size: Max engagements per extraction prompt.
        judge_client: LLM client for IG/IS judge evaluation.
        use_judge: Whether to use LLM judge for IG/IS.
        use_taxonomy: Whether to use interest category mapping.
        taxonomy_client: LLM client for taxonomy mapping.
        taxonomy: Taxonomy object with category IDs. If None, uses bundled taxonomy.
        taxonomy_cache: Optional SQLite-backed cache for taxonomy mappings.
        results_store: Optional ResultsStore to persist IG/IS results and
            taxonomy mappings.  Required when oracle is None.
        test_set_size: IS test set size (backing + distractors).
        max_backing: Max backing items sampled per interest for IS.
        pool_size: Max objects sampled for IS distractor pool.
    """
    if oracle is None and results_store is None:
        raise ValueError(
            "When oracle is None, a results_store (--results-db) is "
            "required to persist IG/IS results for later scoring."
        )
    config = detect_dataset_config(df, dataset_name)

    # Build global pool for IS distractors (all objects across all users)
    pool_df = df.drop_duplicates(subset=["object_id"])

    scores = []
    for user_id in user_ids:
        result = evaluate_user(
            client, df, user_id, config, oracle, model_name, chunk_size,
            judge_client=judge_client, use_judge=use_judge,
            use_taxonomy=use_taxonomy, taxonomy_client=taxonomy_client,
            taxonomy=taxonomy,
            taxonomy_cache=taxonomy_cache, pool_df=pool_df,
            test_set_size=test_set_size, max_backing=max_backing,
            pool_size=pool_size,
        )

        # Persist results to store
        if results_store is not None:
            results_store.save_ig_results(result.ig_results)
            results_store.save_is_results(result.is_results)
            if result.category_map:
                results_store.save_category_map(result.category_map)

        scores.append(result.score)
        logger.info(
            "User %s: IG_F1=%.3f IS=%.3f (oracle=%d)",
            user_id,
            result.score.ig_normalized,
            result.score.is_normalized,
            result.score.oracle_count,
        )

    return scores


