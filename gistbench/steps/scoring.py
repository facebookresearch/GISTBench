"""Scoring: compute per-user GISTBench scores from IG and IS results.

Implements the aggregation formulas from paper Section "Score Aggregation":

Without taxonomy normalization (interests as their own categories):
  - IG_P = fraction of interests that are verified
  - IG_R = sum of verified / oracle_count (oracle = global cross-model count)
  - IG_F1 = harmonic mean of IG_P and IG_R
  - IS = average (correct / backing) over VERIFIED interests only

With taxonomy normalization (interests grouped by category_id):
  - G_c = verified_count / total_count per category
  - S_c = sum(correct/backing) / total_count per category
  - IG_P = sum(G_c) / |categories|
  - IG_R = sum(G_c) / oracle_count
  - IG_F1 = harmonic mean of IG_P and IG_R
  - IS = average S_c over categories with G_c > 0
"""

from __future__ import annotations

from collections import defaultdict

from gistbench.schema import IGResult, ISResult, UserScore


def compute_user_score(
    ig_results: list[IGResult],
    is_results: list[ISResult],
    user_id: str,
    dataset: str,
    model: str,
    oracle_count: int,
    category_map: dict[str, int] | None = None,
) -> UserScore:
    """Compute the final GISTBench score for a user.

    Args:
        ig_results: Per-interest verification results.
        is_results: Per-interest specificity results.
        user_id: User identifier.
        dataset: Dataset name.
        model: Model name.
        oracle_count: Oracle count — number of interest categories with ≥1
            verified interest across all evaluated models for this user.
            This is REQUIRED.
        category_map: Optional mapping from interest name to category_id (int).
            If None, each interest is its own category.
    """
    if not ig_results:
        return UserScore(
            user_id=user_id, dataset=dataset, model=model,
        )

    # Build interest -> verified mapping
    verified_map = {r.interest: r.verified for r in ig_results}

    # Build interest -> IS counts mapping
    is_map = {}
    for r in is_results:
        is_map[r.interest] = (r.correct, r.backing)

    if category_map is None:
        # No taxonomy: each interest is its own category
        num_categories = len(ig_results)
        sum_g_c = sum(1 for r in ig_results if r.verified)

        # IG Precision
        ig_p = sum_g_c / num_categories if num_categories > 0 else 0.0

        # IG Recall
        ig_r = sum_g_c / oracle_count if oracle_count > 0 else 0.0

        # IS Verified: only average over verified interests
        is_ratios = []
        for r in ig_results:
            if r.verified and r.interest in is_map:
                correct, backing = is_map[r.interest]
                if backing > 0:
                    is_ratios.append(correct / backing)
                else:
                    is_ratios.append(0.0)
        is_score = sum(is_ratios) / len(is_ratios) if is_ratios else 0.0

    else:
        # Taxonomy: group interests by category_id
        cat_interests: dict[int, list[str]] = defaultdict(list)
        for interest_name in verified_map:
            cat_id = category_map.get(interest_name, 0)
            cat_interests[cat_id].append(interest_name)

        num_categories = len(cat_interests)

        # Per-category groundedness ratio G_c
        sum_g_c = 0.0
        verified_categories: list[int] = []
        for cat_id, interests in cat_interests.items():
            verified_count = sum(
                1 for i in interests if verified_map.get(i, False)
            )
            g_c = verified_count / len(interests)
            sum_g_c += g_c
            if g_c > 0:
                verified_categories.append(cat_id)

        # IG Precision
        ig_p = sum_g_c / num_categories if num_categories > 0 else 0.0

        # IG Recall
        ig_r = sum_g_c / oracle_count if oracle_count > 0 else 0.0

        # Per-category S_c = sum(correct/backing) / count for ALL interests in category.
        # IS Verified = mean(S_c) over verified categories only (bounded [0,1]).
        cat_s_c: dict[int, float] = {}
        for cat_id, interests in cat_interests.items():
            count = len(interests)
            precise_sum = 0.0
            for i in interests:
                correct, backing = is_map.get(i, (0, 0))
                if backing > 0:
                    precise_sum += correct / backing
            cat_s_c[cat_id] = precise_sum / count if count > 0 else 0.0

        is_ratios = [cat_s_c[c] for c in verified_categories if c in cat_s_c]
        is_score = sum(is_ratios) / len(is_ratios) if is_ratios else 0.0

    # IG F1 (harmonic mean of IG_P and IG_R)
    if ig_p + ig_r > 0:
        ig_f1 = 2 * ig_p * ig_r / (ig_p + ig_r)
    else:
        ig_f1 = 0.0

    # Final score: harmonic mean of IG_F1 and IS
    if ig_f1 + is_score > 0:
        hm = 2 * ig_f1 * is_score / (ig_f1 + is_score)
    else:
        hm = 0.0

    return UserScore(
        user_id=user_id,
        dataset=dataset,
        model=model,
        ig_normalized=ig_f1,
        is_normalized=is_score,
        harmonic_mean=hm,
        oracle_count=oracle_count,
    )
