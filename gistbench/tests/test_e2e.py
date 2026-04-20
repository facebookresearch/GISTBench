"""End-to-end tests for GISTBench pipeline with real LLM inference.

These tests require a valid OPENAI_API_KEY environment variable.
Run with: pytest gistbench/tests/test_e2e.py -v -s

To skip these tests (e.g. in CI without a key), run:
    pytest -m "not e2e"

To run multi-model comparison:
    pytest -m e2e -k "multi_model" -v -s
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from gistbench.client import OpenAIClient
from gistbench.data import detect_dataset_config
from gistbench.download import download_dataset, load_mock_dataset
from gistbench.schema import Oracle, load_mock_oracle
from gistbench.steps.pipeline import evaluate_user, run_benchmark


def _make_test_oracle(user_ids: list[str]) -> Oracle:
    """Create a test oracle with plausible category IDs per user."""
    # Generous oracle so tests don't fail due to strict recall denominator
    # IDs: 192=Music Tutorials, 133=Hiking, 97=Extreme Sports, 16=AI, 86=Drawing
    category_ids = {
        uid: [16, 86, 97, 133, 192]
        for uid in user_ids
    }
    return Oracle(category_ids=category_ids)

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e

# Skip entire module if no API key is set
requires_api_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


def _make_sample_data() -> pd.DataFrame:
    """Create a small synthetic dataset for e2e testing."""
    return pd.DataFrame(
        {
            "user_id": [
                "u1", "u1", "u1", "u1", "u1",
                "u1", "u1", "u1", "u1", "u1",
            ],
            "object_id": [
                "o1", "o2", "o3", "o4", "o5",
                "o6", "o7", "o8", "o9", "o10",
            ],
            "object_text": [
                "Guitar tutorial for beginners, learn basic chords and strumming patterns",
                "Advanced piano techniques, jazz improvisation and chord voicings",
                "Music theory explained: scales, modes, and harmony fundamentals",
                "Best hiking trails in Colorado, mountain scenery and wildlife",
                "Rock climbing gear review: ropes, harnesses, and carabiners",
                "Camping tips for winter: staying warm and safe outdoors",
                "How to bake sourdough bread from scratch, fermentation guide",
                "Italian pasta recipes: carbonara, amatriciana, and cacio e pepe",
                "Drum lessons: rhythm patterns, fills, and groove techniques",
                "Trail running in national parks, ultra marathon preparation",
            ],
            "interaction_type": [
                "explicit_positive",
                "explicit_positive",
                "explicit_positive",
                "explicit_positive",
                "explicit_positive",
                "implicit_positive",
                "implicit_negative",
                "implicit_positive",
                "explicit_positive",
                "implicit_positive",
            ],
            "interaction_time": [
                "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04",
                "2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08",
                "2024-01-09", "2024-01-10",
            ],
        }
    )


@requires_api_key
def test_e2e_single_user():
    """Run full pipeline for a single user with LLM judge."""
    df = _make_sample_data()
    config = detect_dataset_config(df, dataset_name="synthetic")
    client = OpenAIClient(model="gpt-4o-mini")

    oracle = _make_test_oracle(["u1"])
    result = evaluate_user(
        client=client,
        df=df,
        user_id="u1",
        config=config,
        oracle=oracle,
        model_name="gpt-4o-mini",
        use_judge=True,
    )
    score = result.score

    assert score.user_id == "u1"
    assert score.dataset == "synthetic"
    assert score.model == "gpt-4o-mini"
    assert 0.0 <= score.ig_normalized <= 1.0
    assert 0.0 <= score.is_normalized <= 1.0
    assert score.oracle_count > 0, "Expected at least one interest to be extracted"

    print(f"\n--- E2E Single User Results (with judge) ---")
    print(f"  Interests found: {score.oracle_count}")
    print(f"  IG_F1:  {score.ig_normalized:.3f}")
    print(f"  IS:     {score.is_normalized:.3f}")


@requires_api_key
def test_e2e_run_benchmark():
    """Run the full benchmark across multiple users."""
    df = _make_sample_data()

    # Add a second user
    user2 = pd.DataFrame(
        {
            "user_id": ["u2"] * 6,
            "object_id": ["o20", "o21", "o22", "o23", "o24", "o25"],
            "object_text": [
                "Python programming tutorial: data structures and algorithms",
                "Machine learning with PyTorch: neural networks from scratch",
                "Deep learning for computer vision: CNNs and object detection",
                "Basketball highlights: NBA playoffs best dunks and plays",
                "Soccer tactics explained: formations and pressing strategies",
                "Statistics for data science: hypothesis testing and regression",
            ],
            "interaction_type": [
                "explicit_positive",
                "explicit_positive",
                "explicit_positive",
                "implicit_positive",
                "implicit_negative",
                "explicit_positive",
            ],
            "interaction_time": [
                "2024-02-01", "2024-02-02", "2024-02-03",
                "2024-02-04", "2024-02-05", "2024-02-06",
            ],
        }
    )
    df = pd.concat([df, user2], ignore_index=True)

    oracle = _make_test_oracle(["u1", "u2"])
    client = OpenAIClient(model="gpt-4o-mini")
    scores = run_benchmark(
        client=client,
        df=df,
        user_ids=["u1", "u2"],
        oracle=oracle,
        dataset_name="synthetic",
        model_name="gpt-4o-mini",
        use_judge=True,
    )

    assert len(scores) == 2

    for score in scores:
        assert score.model == "gpt-4o-mini"
        assert 0.0 <= score.ig_normalized <= 1.0
        assert 0.0 <= score.is_normalized <= 1.0
        assert score.oracle_count > 0

    print(f"\n--- E2E Benchmark Results ---")
    for s in scores:
        print(
            f"  User {s.user_id}: IG_F1={s.ig_normalized:.3f} "
            f"IS={s.is_normalized:.3f} "
            f"interests={s.oracle_count}"
        )


@requires_api_key
def test_e2e_mock_dataset_download():
    """Test loading the bundled mock dataset and running evaluation."""
    df = load_mock_dataset()
    assert len(df) > 0
    assert "user_id" in df.columns

    oracle = load_mock_oracle()
    client = OpenAIClient(model="gpt-4o-mini")
    user_ids = sorted(df["user_id"].unique())[:1]

    scores = run_benchmark(
        client=client,
        df=df,
        user_ids=user_ids,
        oracle=oracle,
        dataset_name="synthetic",
        model_name="gpt-4o-mini",
        use_judge=True,
    )

    assert len(scores) == 1
    assert scores[0].oracle_count > 0

    print(f"\n--- Mock Dataset Results ---")
    s = scores[0]
    print(
        f"  User {s.user_id}: IG_F1={s.ig_normalized:.3f} "
        f"IS={s.is_normalized:.3f} "
        f"interests={s.oracle_count}"
    )


@requires_api_key
@pytest.mark.parametrize("model_name", ["gpt-4o-mini", "gpt-4o"])
def test_e2e_multi_model(model_name):
    """Compare metrics across different GPT model versions.

    Run with: pytest -m e2e -k "multi_model" -v -s
    """
    df = _make_sample_data()

    # Add second user for broader coverage
    user2 = pd.DataFrame(
        {
            "user_id": ["u2"] * 6,
            "object_id": ["o20", "o21", "o22", "o23", "o24", "o25"],
            "object_text": [
                "Python programming tutorial: data structures and algorithms",
                "Machine learning with PyTorch: neural networks from scratch",
                "Deep learning for computer vision: CNNs and object detection",
                "Basketball highlights: NBA playoffs best dunks and plays",
                "Soccer tactics explained: formations and pressing strategies",
                "Statistics for data science: hypothesis testing and regression",
            ],
            "interaction_type": [
                "explicit_positive",
                "explicit_positive",
                "explicit_positive",
                "implicit_positive",
                "implicit_negative",
                "explicit_positive",
            ],
            "interaction_time": [
                "2024-02-01", "2024-02-02", "2024-02-03",
                "2024-02-04", "2024-02-05", "2024-02-06",
            ],
        }
    )
    df = pd.concat([df, user2], ignore_index=True)

    oracle = _make_test_oracle(["u1", "u2"])
    client = OpenAIClient(model=model_name)
    scores = run_benchmark(
        client=client,
        df=df,
        user_ids=["u1", "u2"],
        oracle=oracle,
        dataset_name="synthetic",
        model_name=model_name,
        use_judge=True,
    )

    assert len(scores) == 2
    for score in scores:
        assert score.model == model_name
        assert 0.0 <= score.ig_normalized <= 1.0
        assert 0.0 <= score.is_normalized <= 1.0
        assert score.oracle_count > 0

    avg_ig = sum(s.ig_normalized for s in scores) / len(scores)
    avg_is = sum(s.is_normalized for s in scores) / len(scores)
    total_interests = sum(s.oracle_count for s in scores)

    print(f"\n--- {model_name} Results ---")
    for s in scores:
        print(
            f"  User {s.user_id}: IG_F1={s.ig_normalized:.3f} "
            f"IS={s.is_normalized:.3f} "
            f"interests={s.oracle_count}"
        )
    print(
        f"  AVG: IG_F1={avg_ig:.3f} IS={avg_is:.3f} "
        f"total_interests={total_interests}"
    )
