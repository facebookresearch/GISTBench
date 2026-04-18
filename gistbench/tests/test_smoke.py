"""Integration tests for GISTBench — no API key required.

These tests exercise real code paths end-to-end using mock LLM clients
and bundled assets. They cover the full pipeline from data loading through
scoring, without making external API calls.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from gistbench.client import parse_json_response
from gistbench.data import chunk_user_history, detect_dataset_config, load_dataset, sample_users
from gistbench.prompts.interest_extraction import (
    build_extraction_messages,
    build_extraction_prompt,
    format_engagement_history,
)
from gistbench.schema import (
    DATASET_CONFIGS,
    IGResult,
    ISResult,
    Interest,
    Oracle,
    UserScore,
    load_mock_oracle,
)
from gistbench.steps.interest_groundedness import evaluate_ig, verify_interest
from gistbench.steps.interest_specificity import evaluate_is
from gistbench.steps.pipeline import evaluate_user, parse_interests, run_benchmark
from gistbench.steps.scoring import compute_user_score
from gistbench.steps.taxonomy import Taxonomy, load_default_taxonomy
from gistbench.store import ResultsStore


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def user_df():
    """Realistic multi-signal engagement DataFrame for two users."""
    return pd.DataFrame(
        {
            "user_id": ["u1"] * 8 + ["u2"] * 5,
            "object_id": [f"o{i}" for i in range(1, 14)],
            "object_text": [
                "Guitar tutorial for beginners",
                "Advanced guitar solos and techniques",
                "Blues guitar licks and riffs",
                "Hiking trails in Colorado mountains",
                "Best camping gear for winter",
                "Trail running ultra marathon prep",
                "Boring infomercial about kitchen tools",
                "Random news segment about weather",
                "Python programming data structures",
                "Machine learning with PyTorch",
                "Deep learning computer vision CNNs",
                "Basketball NBA highlights dunks",
                "Soccer tactics formations pressing",
            ],
            "interaction_type": [
                "explicit_positive",
                "explicit_positive",
                "implicit_positive",
                "explicit_positive",
                "implicit_positive",
                "implicit_positive",
                "implicit_negative",
                "implicit_negative",
                "explicit_positive",
                "explicit_positive",
                "explicit_positive",
                "implicit_positive",
                "implicit_negative",
            ],
        }
    )


def _mock_client(responses):
    """Create a mock LLM client returning pre-defined responses in order."""
    client = MagicMock()
    client.chat = MagicMock(side_effect=responses)
    return client


# ---------------------------------------------------------------------------
# 1. Data loading → config detection → prompt generation (full chain)
# ---------------------------------------------------------------------------


def test_data_to_prompt_chain(user_df):
    """Load data → detect config → chunk history → build prompt → format engagements."""
    config = detect_dataset_config(user_df)
    assert config.has_explicit_positive
    assert config.has_implicit_positive
    assert config.has_implicit_negative
    assert not config.has_explicit_negative

    chunks = chunk_user_history(user_df, "u1", chunk_size=4)
    assert len(chunks) == 2
    assert len(chunks[0]) + len(chunks[1]) == 8

    prompt = build_extraction_prompt(config)
    assert config.object_name in prompt
    assert "3 implicit_positive" in prompt
    assert "2 explicit_positive" in prompt

    engagements = chunks[0].to_dict("records")
    messages = build_extraction_messages(config, engagements)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert engagements[0]["object_text"] in messages[0]["content"]


def test_named_dataset_configs_generate_valid_prompts():
    """Every built-in dataset config produces a valid extraction prompt."""
    for name, cfg in DATASET_CONFIGS.items():
        prompt = build_extraction_prompt(cfg)
        assert len(prompt) > 100, f"Prompt too short for {name}"
        assert cfg.object_name in prompt


def test_sample_users_deterministic(user_df):
    """Sampling users is deterministic and respects seed."""
    s1 = sample_users(user_df, n=1, seed=42)
    s2 = sample_users(user_df, n=1, seed=42)
    assert s1 == s2
    assert len(s1) == 1


# ---------------------------------------------------------------------------
# 2. JSON parsing from LLM responses (various formats)
# ---------------------------------------------------------------------------


def test_parse_json_various_formats():
    """parse_json_response handles raw, fenced, embedded, object, and invalid."""
    assert parse_json_response('[{"a": 1}]') == [{"a": 1}]
    assert parse_json_response('```json\n[{"b": 2}]\n```') == [{"b": 2}]
    assert parse_json_response('Here is: [{"c": 3}] done') == [{"c": 3}]
    assert parse_json_response('{"key": "val"}') == {"key": "val"}
    assert parse_json_response("not json at all") is None


# ---------------------------------------------------------------------------
# 3. IG verification → scoring pipeline (no LLM judge)
# ---------------------------------------------------------------------------


def test_ig_verification_with_thresholds(user_df):
    """verify_interest uses dataset thresholds correctly on real engagement data."""
    config = detect_dataset_config(user_df, dataset_name="synthetic")
    u1_df = user_df[user_df["user_id"] == "u1"]

    # Guitar: o1 (explicit+), o2 (explicit+), o3 (implicit+) → meets 2 explicit+ threshold
    guitar = Interest(name="Guitar", item_ids=["o1", "o2", "o3"])
    assert verify_interest(guitar, ["o1", "o2", "o3"], u1_df, config) is True

    # Hiking with only 1 explicit+ and 1 implicit+ → doesn't meet any threshold
    hiking_weak = Interest(name="Hiking", item_ids=["o4"])
    assert verify_interest(hiking_weak, ["o4"], u1_df, config) is False

    # Empty evidence → not verified
    empty = Interest(name="Nothing", item_ids=[])
    assert verify_interest(empty, [], u1_df, config) is False


def test_ig_evaluation_without_judge(user_df):
    """evaluate_ig without a judge client uses all cited items for verification."""
    config = detect_dataset_config(user_df, dataset_name="synthetic")
    u1_df = user_df[user_df["user_id"] == "u1"]

    interests = [
        Interest(name="Guitar", item_ids=["o1", "o2", "o3"]),
        Interest(name="Hiking", item_ids=["o4"]),
    ]
    results = evaluate_ig(interests, "u1", u1_df, config, "test-model", judge_client=None)
    assert len(results) == 2
    guitar_r = next(r for r in results if r.interest == "Guitar")
    hiking_r = next(r for r in results if r.interest == "Hiking")
    assert guitar_r.verified is True
    assert hiking_r.verified is False


# ---------------------------------------------------------------------------
# 4. Full scoring pipeline: IG + IS → category grouping → harmonic mean
# ---------------------------------------------------------------------------


def test_scoring_without_taxonomy():
    """Scoring without taxonomy: each interest = own category, HM(IG_F1, IS)."""
    ig = [
        IGResult(user_id="u1", dataset="d", model="m", interest="Guitar", verified=True),
        IGResult(user_id="u1", dataset="d", model="m", interest="Hiking", verified=True),
        IGResult(user_id="u1", dataset="d", model="m", interest="Bogus", verified=False),
    ]
    is_ = [
        ISResult(user_id="u1", dataset="d", model="m", interest="Guitar", correct=4, selected=4, backing=5),
        ISResult(user_id="u1", dataset="d", model="m", interest="Hiking", correct=3, selected=3, backing=5),
        ISResult(user_id="u1", dataset="d", model="m", interest="Bogus", correct=0, selected=0, backing=0),
    ]
    score = compute_user_score(ig, is_, "u1", "d", "m", oracle_count=3)

    # IG_P = 2/3, IG_R = 2/3, IG_F1 = 2/3
    assert score.ig_normalized == pytest.approx(2 / 3)
    # IS = mean(4/5, 3/5) over verified only = 0.7
    assert score.is_normalized == pytest.approx(0.7)
    # HM(2/3, 0.7)
    expected_hm = 2 * (2 / 3) * 0.7 / ((2 / 3) + 0.7)
    assert score.harmonic_mean == pytest.approx(expected_hm)
    assert score.oracle_count == 3


def test_scoring_with_taxonomy():
    """Scoring with taxonomy: interests grouped by category, G_c/S_c per category."""
    ig = [
        IGResult(user_id="u1", dataset="d", model="m", interest="guitar", verified=True),
        IGResult(user_id="u1", dataset="d", model="m", interest="piano", verified=True),
        IGResult(user_id="u1", dataset="d", model="m", interest="hiking", verified=True),
    ]
    is_ = [
        ISResult(user_id="u1", dataset="d", model="m", interest="guitar", correct=4, selected=4, backing=5),
        ISResult(user_id="u1", dataset="d", model="m", interest="piano", correct=3, selected=3, backing=5),
        ISResult(user_id="u1", dataset="d", model="m", interest="hiking", correct=5, selected=5, backing=5),
    ]
    category_map = {"guitar": 1, "piano": 1, "hiking": 2}
    score = compute_user_score(ig, is_, "u1", "d", "m", oracle_count=3, category_map=category_map)

    # 2 categories, all verified → G_c=1 for both → sum_G_c=2
    # IG_P = 2/2 = 1.0, IG_R = 2/3
    # IS: cat1 S_c = (4/5 + 3/5)/2 = 0.7, cat2 S_c = 5/5 = 1.0 → mean = 0.85
    assert score.ig_normalized > 0
    assert score.is_normalized > 0
    assert score.harmonic_mean > 0
    assert score.harmonic_mean < 1.0  # Not perfect


def test_scoring_empty_inputs():
    """Empty IG results → all zeros."""
    score = compute_user_score([], [], "u1", "d", "m", oracle_count=5)
    assert score.ig_normalized == 0.0
    assert score.is_normalized == 0.0
    assert score.harmonic_mean == 0.0


# ---------------------------------------------------------------------------
# 5. Full pipeline with mock LLM client (extraction → IG → IS → scoring)
# ---------------------------------------------------------------------------


def test_evaluate_user_with_mock_client(user_df):
    """Full pipeline: extraction → IG → IS → scoring using mock LLM responses."""
    config = detect_dataset_config(user_df, dataset_name="synthetic")
    oracle = Oracle(category_ids={"u1": [1, 2, 3]})

    # Mock LLM: extraction returns interests, IG judge returns relevant indices,
    # IS stage1 returns "NONE", IS judge returns item IDs
    extraction_response = json.dumps([
        {"interest": "Guitar Music", "item_ids": ["o1", "o2", "o3"], "evidence_excerpt": "guitar"},
        {"interest": "Outdoor Hiking", "item_ids": ["o4", "o5", "o6"], "evidence_excerpt": "hiking"},
    ])
    ig_judge_response_1 = "0, 1, 2"  # All relevant for Guitar
    ig_judge_response_2 = "0, 1, 2"  # All relevant for Hiking
    is_stage1_response = "NONE"  # No pool overlap
    is_judge_response_1 = "item_0, item_1"  # IS judge for Guitar
    is_judge_response_2 = "item_0"  # IS judge for Hiking

    mock = _mock_client([
        extraction_response,      # extract interests
        ig_judge_response_1,       # IG judge for Guitar
        ig_judge_response_2,       # IG judge for Hiking
        is_stage1_response,        # IS stage 1 shortlisting
        is_judge_response_1,       # IS judge for Guitar
        is_judge_response_2,       # IS judge for Hiking
    ])

    result = evaluate_user(
        client=mock, df=user_df, user_id="u1", config=config,
        oracle=oracle, model_name="mock", use_judge=True, use_taxonomy=False,
    )

    assert result.score.user_id == "u1"
    assert result.score.model == "mock"
    assert result.score.oracle_count == 3
    assert len(result.ig_results) == 2
    assert len(result.is_results) == 2
    # Pipeline completed without errors — scores are valid floats
    assert 0.0 <= result.score.ig_normalized <= 1.0
    assert 0.0 <= result.score.is_normalized <= 1.0


def test_evaluate_user_no_judge(user_df):
    """Pipeline without judge: extraction → threshold verification → simple IS ratio."""
    config = detect_dataset_config(user_df, dataset_name="synthetic")
    oracle = Oracle(category_ids={"u1": [1, 2]})

    extraction_response = json.dumps([
        {"interest": "Guitar Music", "item_ids": ["o1", "o2", "o3"]},
    ])
    mock = _mock_client([extraction_response])

    result = evaluate_user(
        client=mock, df=user_df, user_id="u1", config=config,
        oracle=oracle, model_name="mock", use_judge=False, use_taxonomy=False,
    )

    assert len(result.ig_results) == 1
    assert len(result.is_results) == 1
    # Without judge, IS is not evaluated (all zeros)
    assert result.is_results[0].backing == 0
    assert result.is_results[0].correct == 0
    assert result.is_results[0].selected == 0


# ---------------------------------------------------------------------------
# 6. Oracle + Taxonomy integration (bundled assets)
# ---------------------------------------------------------------------------


def test_bundled_taxonomy_loads():
    """Default taxonomy loads all 325 categories with correct structure."""
    tax = load_default_taxonomy()
    assert len(tax) == 325
    assert all(isinstance(k, int) for k in tax.id_to_name)
    assert all(isinstance(v, str) for v in tax.id_to_name.values())
    # Bidirectional mapping is consistent
    for cid, name in tax.id_to_name.items():
        assert tax.name_to_id[name] == cid


def test_mock_oracle_loads_with_int_ids():
    """Mock oracle loads with integer category IDs and correct user count."""
    oracle = load_mock_oracle()
    assert len(oracle.category_ids) == 3
    for uid, ids in oracle.category_ids.items():
        assert all(isinstance(cid, int) for cid in ids)
        assert oracle.count(uid) == len(ids)
    assert oracle.user_ids() == sorted(oracle.category_ids.keys())


def test_oracle_to_file_roundtrip():
    """Oracle.to_file → Oracle.from_file preserves content and dataset key."""
    oracle = Oracle(category_ids={"u1": [1, 2, 5], "u2": [9]})
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tmp_path = f.name
    try:
        oracle.to_file(tmp_path, dataset="synthetic")
        # Verify the on-disk format includes the dataset key
        payload = json.loads(Path(tmp_path).read_text())
        assert payload["dataset"] == "synthetic"
        assert payload["oracle"]["u1"] == [1, 2, 5]
        # Round-trip
        reloaded = Oracle.from_file(tmp_path)
        assert reloaded.category_ids == oracle.category_ids
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def test_export_oracle_unions_store_with_existing_json():
    """Exporting from a store + --merge file produces the per-user union."""
    from click.testing import CliRunner

    from gistbench.cli import main

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        existing_path = f.name
    out_path = Path(tempfile.mktemp(suffix=".json"))

    try:
        # Store: u1 has verified guitar→cat 4 and verified jazz→cat 5
        store = ResultsStore(db_path)
        store.save_ig_results([
            IGResult(user_id="u1", dataset="d", model="m_a", interest="guitar", verified=True),
            IGResult(user_id="u1", dataset="d", model="m_a", interest="jazz", verified=True),
        ])
        store.save_category_map({"guitar": 4, "jazz": 5})
        store.close()

        # Existing oracle: u1 = [1, 2, 3], u2 = [9]
        Path(existing_path).write_text(
            json.dumps({"oracle": {"u1": [1, 2, 3], "u2": [9]}})
        )

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "export-oracle",
                "--results-db", db_path,
                "-d", "d",
                "-o", str(out_path),
                "--merge", existing_path,
            ],
        )
        assert result.exit_code == 0, result.output

        # Reload exported file: u1 = union {1,2,3,4,5}, u2 untouched = [9]
        merged = Oracle.from_file(out_path)
        assert merged.category_ids["u1"] == [1, 2, 3, 4, 5]
        assert merged.category_ids["u2"] == [9]
    finally:
        for p in [db_path, existing_path, out_path]:
            Path(p).unlink(missing_ok=True)


def test_oracle_merge_unions_per_user():
    """Oracle.merge unions category IDs per user; users in either are kept."""
    a = Oracle(category_ids={"u1": [1, 2], "u2": [10]})
    b = Oracle(category_ids={"u1": [2, 3], "u3": [99]})
    merged = a.merge(b)
    assert merged.category_ids["u1"] == [1, 2, 3]
    assert merged.category_ids["u2"] == [10]
    assert merged.category_ids["u3"] == [99]
    # Original oracles are unchanged
    assert a.category_ids["u1"] == [1, 2]
    assert b.category_ids["u1"] == [2, 3]


def test_oracle_from_json_file():
    """Oracle round-trips through JSON correctly."""
    data = {"oracle": {"a": [1, 2, 3], "b": [10, 20]}}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        tmp_path = f.name

    oracle = Oracle.from_file(tmp_path)
    assert oracle.count("a") == 3
    assert oracle.count("b") == 2
    assert oracle.count("missing") == 0
    assert set(oracle.user_ids()) == {"a", "b"}
    Path(tmp_path).unlink()


# ---------------------------------------------------------------------------
# 7. ResultsStore: save → load → rescore cycle
# ---------------------------------------------------------------------------


def test_results_store_full_cycle():
    """Store saves IG/IS/taxonomy, retrieves them, and rescores correctly."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        store = ResultsStore(db_path)

        # Save IG + IS results for three models (min required for scoring)
        for model in ["model_a", "model_b", "model_c"]:
            ig = [
                IGResult(user_id="u1", dataset="test", model=model, interest="guitar", verified=True),
                IGResult(user_id="u1", dataset="test", model=model, interest="hiking", verified=model == "model_a"),
            ]
            is_ = [
                ISResult(user_id="u1", dataset="test", model=model, interest="guitar", correct=4, selected=4, backing=5),
                ISResult(user_id="u1", dataset="test", model=model, interest="hiking", correct=3, selected=3, backing=5),
            ]
            store.save_ig_results(ig)
            store.save_is_results(is_)

        # Save taxonomy
        store.save_category_map({"guitar": 1, "hiking": 2})

        # Verify retrieval
        assert store.list_models("test") == ["model_a", "model_b", "model_c"]
        assert store.list_users("test") == ["u1"]
        assert store.get_category_map() == {"guitar": 1, "hiking": 2}

        ig_loaded = store.get_ig_results("model_a", "test", "u1")
        assert len(ig_loaded) == 2
        assert any(r.verified for r in ig_loaded)

        # Rescore requires 3+ models (default min_models=3)
        scores = store.rescore_all("test")
        assert len(scores) == 3  # One per model
        for s in scores:
            assert s.user_id == "u1"
            assert s.dataset == "test"
            assert 0.0 <= s.ig_normalized <= 1.0
            assert 0.0 <= s.is_normalized <= 1.0
            assert 0.0 <= s.harmonic_mean <= 1.0

        # Fewer than min_models raises
        with pytest.raises(ValueError, match="min_models"):
            store.rescore_all("test", min_models=10)

        # Cross-model oracle: hiking verified by model_a → oracle sees 2 categories
        oracle_count = store.compute_oracle("u1", "test")
        assert oracle_count == 2

        info = store.summary("test")
        assert info["ig_results"] == 6  # 2 interests × 3 models
        assert info["is_results"] == 6
        assert info["taxonomy_mappings"] == 2

        store.close()
    finally:
        Path(db_path).unlink(missing_ok=True)


def test_results_store_export_import_csv():
    """Store export → import round-trip preserves data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db1_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db2_path = f.name
    metrics_csv = Path(tempfile.mktemp(suffix=".csv"))
    taxonomy_csv = Path(tempfile.mktemp(suffix=".csv"))

    try:
        # Create and populate store 1
        store1 = ResultsStore(db1_path)
        store1.save_ig_results([
            IGResult(user_id="u1", dataset="d", model="m", interest="cooking", verified=True),
        ])
        store1.save_is_results([
            ISResult(user_id="u1", dataset="d", model="m", interest="cooking", correct=3, selected=4, backing=5),
        ])
        store1.save_category_map({"cooking": 42})

        # Export
        n_metrics = store1.export_metrics_csv(metrics_csv)
        n_taxonomy = store1.export_taxonomy_csv(taxonomy_csv)
        assert n_metrics == 1
        assert n_taxonomy == 1
        store1.close()

        # Import into store 2
        store2 = ResultsStore(db2_path)
        store2.import_metrics_csv(metrics_csv)
        store2.import_taxonomy_csv(taxonomy_csv)

        ig = store2.get_ig_results("m", "d", "u1")
        assert len(ig) == 1
        assert ig[0].verified is True

        cat = store2.get_category("cooking")
        assert cat == 42
        store2.close()
    finally:
        for p in [db1_path, db2_path, metrics_csv, taxonomy_csv]:
            Path(p).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 8. Data loading from files
# ---------------------------------------------------------------------------


def test_load_dataset_csv():
    """load_dataset reads CSV with required columns."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("user_id,object_id,object_text,interaction_type\n")
        f.write("u1,o1,hello world,explicit_positive\n")
        f.write("u1,o2,goodbye world,implicit_negative\n")
        tmp_path = f.name

    try:
        df = load_dataset(tmp_path)
        assert len(df) == 2
        assert set(df.columns) >= {"user_id", "object_id", "object_text", "interaction_type"}
        config = detect_dataset_config(df)
        assert config.has_explicit_positive
        assert config.has_implicit_negative
    finally:
        Path(tmp_path).unlink()


# ---------------------------------------------------------------------------
# 9. run_benchmark with mock client (multi-user pipeline)
# ---------------------------------------------------------------------------


def test_run_benchmark_multi_user(user_df):
    """run_benchmark processes multiple users end-to-end with mock client."""
    oracle = Oracle(category_ids={"u1": [1, 2], "u2": [3, 4]})

    # Each user gets: extraction call, then no-judge path
    extraction_u1 = json.dumps([
        {"interest": "Guitar", "item_ids": ["o1", "o2", "o3"]},
    ])
    extraction_u2 = json.dumps([
        {"interest": "Programming", "item_ids": ["o9", "o10", "o11"]},
    ])
    mock = _mock_client([extraction_u1, extraction_u2])

    scores = run_benchmark(
        client=mock, df=user_df, user_ids=["u1", "u2"], oracle=oracle,
        dataset_name="synthetic", model_name="mock",
        use_judge=False, use_taxonomy=False,
    )

    assert len(scores) == 2
    assert {s.user_id for s in scores} == {"u1", "u2"}
    for s in scores:
        assert s.model == "mock"
        assert s.dataset == "synthetic"
        assert 0.0 <= s.ig_normalized <= 1.0
        assert 0.0 <= s.is_normalized <= 1.0
