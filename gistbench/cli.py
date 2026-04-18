"""GISTBench command-line interface."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import click

from gistbench.client import OpenAIClient
from gistbench.data import load_dataset, sample_users, validate_dataset
from gistbench.download import download_dataset, load_mock_dataset
from gistbench.schema import (
    DATASET_CONFIGS,
    Oracle,
    UserScore,
    load_bundled_oracle,
    load_mock_oracle,
)
from gistbench.steps.pipeline import run_benchmark
from gistbench.steps.taxonomy import TaxonomyCache
from gistbench.store import ResultsStore

MIN_ORACLE_MODELS = 3


def _generate_report(
    scores: list[UserScore],
    dataset_name: str,
    title: str = "GISTBench Results",
) -> str:
    """Generate a markdown report from scores."""
    lines = [f"# {title}", ""]
    lines.append(f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Dataset:** {dataset_name}")

    models = sorted({s.model for s in scores})
    users = sorted({s.user_id for s in scores})
    lines.append(f"**Models:** {len(models)}  ")
    lines.append(f"**Users:** {len(users)}")
    lines.append("")

    # Model summary table
    lines.append("## Model Summary")
    lines.append("")
    lines.append("| Model | IG_F1 | IS | HM | Users |")
    lines.append("|---|---|---|---|---|")
    for m in models:
        m_scores = [s for s in scores if s.model == m]
        avg_ig = sum(s.ig_normalized for s in m_scores) / len(m_scores)
        avg_is = sum(s.is_normalized for s in m_scores) / len(m_scores)
        avg_hm = sum(s.harmonic_mean for s in m_scores) / len(m_scores)
        lines.append(f"| {m} | {avg_ig:.3f} | {avg_is:.3f} | {avg_hm:.3f} | {len(m_scores)} |")
    lines.append("")

    # Per-user detail table
    lines.append("## Per-User Scores")
    lines.append("")
    lines.append("| User | Model | IG_F1 | IS | HM | Oracle |")
    lines.append("|---|---|---|---|---|---|")
    for s in sorted(scores, key=lambda s: (s.user_id, s.model)):
        lines.append(
            f"| {s.user_id} | {s.model} | {s.ig_normalized:.3f} "
            f"| {s.is_normalized:.3f} | {s.harmonic_mean:.3f} | {s.oracle_count} |"
        )
    lines.append("")

    return "\n".join(lines)


def _resolve_report_path(
    report: str | None, report_dir: str | None, name: str,
) -> Path | None:
    """Resolve where to write a markdown report.

    Precedence: explicit ``--report PATH`` wins over ``--report-dir DIR``.
    When ``--report-dir`` is used, the file is named
    ``{name}-{UTC-timestamp}.md`` and the directory is created if missing.
    Returns ``None`` when neither flag is set.
    """
    if report:
        return Path(report)
    if report_dir:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        d = Path(report_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{name}-{ts}.md"
    return None


def _load_dotenv() -> None:
    """Load .env file from the current directory if it exists."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
def main(verbose: bool) -> None:
    """GISTBench: Groundedness & Interest Specificity Test Bench."""
    _load_dotenv()
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


@main.command()
@click.option(
    "--dataset",
    "-d",
    default="synthetic",
    help="Dataset name or path to a local data file (CSV/JSON/JSONL).",
)
@click.option("--model", "-m", default="gpt-4o-mini", help="Model name.")
@click.option("--base-url", default=None, help="OpenAI-compatible API base URL.")
@click.option("--api-key", default=None, help="API key (defaults to OPENAI_API_KEY env var).")
@click.option("--num-users", "-n", default=0, help="Number of users to evaluate (0 = all).")
@click.option("--seed", default=42, help="Random seed for user sampling.")
@click.option("--output", "-o", default=None, help="Output JSON file for results.")
@click.option("--no-judge", is_flag=True, help="Disable LLM judge for IG/IS evaluation (faster, less accurate).")
@click.option("--no-taxonomy", is_flag=True, help="Disable interest category mapping (each interest is its own category).")
@click.option("--taxonomy-cache", default=None, help="Path to SQLite file for caching taxonomy mappings (e.g., taxonomy_cache.db).")
@click.option("--results-db", required=True, help="Path to SQLite file for persisting all results.")
@click.option("--oracle", "oracle_path", default=None, help="Path to pre-computed oracle JSON file. Merged with store contributions. If not provided, the bundled oracle is used for 'synthetic'; otherwise predictions from 3+ models in --results-db.")
@click.option("--chunk-size", default=100, show_default=True, help="Max engagements per extraction prompt.")
@click.option("--test-set-size", default=50, show_default=True, help="IS test set size (backing + distractors).")
@click.option("--max-backing", default=5, show_default=True, help="Max backing items sampled per interest for IS.")
@click.option("--pool-size", default=1000, show_default=True, help="Max objects sampled for IS distractor pool.")
@click.option("--mock", is_flag=True, help="Use the bundled mock dataset (3 users) instead of downloading.")
@click.option("--report", default=None, help="Explicit path to write a markdown report (overrides --report-dir).")
@click.option("--report-dir", default=None, help="Directory for timestamped markdown reports (e.g., 'reports'). Created if missing.")
def run(
    dataset: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    num_users: int,
    seed: int,
    output: str | None,
    no_judge: bool,
    no_taxonomy: bool,
    taxonomy_cache: str | None,
    results_db: str,
    oracle_path: str | None,
    chunk_size: int,
    test_set_size: int,
    max_backing: int,
    pool_size: int,
    mock: bool,
    report: str | None,
    report_dir: str | None,
) -> None:
    """Run the GISTBench evaluation pipeline.

    Every run saves IG/IS predictions to --results-db. Oracle resolution:

      1. ``--oracle path.json``  → load file, union with store contributions.
      2. dataset == "synthetic" → load bundled oracle, union with store.
      3. otherwise              → require 3+ models in the store and union
                                  their verified interest categories.

    The 3-model gate only applies to case (3); cases (1) and (2) score
    immediately after a single run.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        click.echo("Error: No API key provided. Set OPENAI_API_KEY or use --api-key.", err=True)
        sys.exit(1)

    # Load dataset
    if mock:
        click.echo("Loading bundled mock dataset (3 users, 25 engagements).")
        df = load_mock_dataset()
        dataset_name = "mock"
    else:
        data_path = Path(dataset)
        if data_path.exists():
            click.echo(f"Loading dataset from file: {dataset}")
            df = load_dataset(data_path)
            dataset_name = data_path.stem
        else:
            click.echo(f"Downloading dataset: {dataset}")
            df = download_dataset(dataset)
            dataset_name = dataset

    validate_dataset(df)

    # Sample users
    all_users = sorted(df["user_id"].unique())
    if num_users > 0:
        user_ids = sample_users(df, n=num_users, seed=seed)
    else:
        user_ids = all_users

    click.echo(f"Evaluating {len(user_ids)} users with model '{model}'...")

    # Initialize client
    client = OpenAIClient(model=model, api_key=api_key, base_url=base_url)

    # Set up taxonomy cache
    cache = None
    if taxonomy_cache and not no_taxonomy:
        cache = TaxonomyCache(taxonomy_cache)
        click.echo(f"Using taxonomy cache: {taxonomy_cache} ({cache.count()} cached mappings)")

    # Set up results store
    store = ResultsStore(results_db)
    info = store.summary(dataset_name)
    click.echo(f"Results store: {results_db} ({info['ig_results']} IG, {info['is_results']} IS, {info['taxonomy_mappings']} taxonomy)")

    # Run extraction + IG/IS (no scoring yet — oracle comes after)
    run_benchmark(
        client=client,
        df=df,
        user_ids=user_ids,
        oracle=None,
        dataset_name=dataset_name,
        model_name=model,
        chunk_size=chunk_size,
        use_judge=not no_judge,
        use_taxonomy=not no_taxonomy,
        taxonomy_cache=cache,
        results_store=store,
        test_set_size=test_set_size,
        max_backing=max_backing,
        pool_size=pool_size,
    )

    if cache is not None:
        click.echo(f"Taxonomy cache now has {cache.count()} mappings")
        cache.close()

    # --- Scoring ---
    # Predictions are now in the store. Determine oracle and score.

    base_oracle: Oracle | None = None
    if oracle_path:
        base_oracle = Oracle.from_file(oracle_path)
        click.echo(f"\nOracle loaded from {oracle_path} ({len(base_oracle.category_ids)} users)")
    elif dataset_name == "synthetic":
        base_oracle = load_bundled_oracle()
        click.echo(f"\nUsing bundled synthetic oracle ({len(base_oracle.category_ids)} users)")

    if base_oracle is not None:
        # Union the file/bundled oracle with store contributions so newly
        # verified categories from this run augment the denominator.
        store_oracle = Oracle.from_results_store(store, dataset_name)
        oracle = base_oracle.merge(store_oracle)
        click.echo(
            f"Merged with store: {len(oracle.category_ids)} users "
            f"({len(store_oracle.category_ids)} from store)"
        )
        all_scores = store.rescore_with_oracle(dataset_name, oracle)
    else:
        # Build oracle from predictions — require 3+ models
        models = store.list_models(dataset_name)
        click.echo(f"\nModels in store: {len(models)} ({', '.join(models)})")
        if len(models) < MIN_ORACLE_MODELS:
            info = store.summary(dataset_name)
            click.echo(
                f"Need {MIN_ORACLE_MODELS} models to build oracle, "
                f"have {len(models)}. Run {MIN_ORACLE_MODELS - len(models)} more."
            )
            click.echo(f"Results store: {info['ig_results']} IG, {info['is_results']} IS, {info['taxonomy_mappings']} taxonomy")
            store.close()
            sys.exit(0)

        click.echo(f"Building oracle from {len(models)} models and scoring...")
        all_scores = store.rescore_all(dataset_name, min_models=MIN_ORACLE_MODELS)

    info = store.summary(dataset_name)
    click.echo(f"Results store: {info['ig_results']} IG, {info['is_results']} IS, {info['taxonomy_mappings']} taxonomy")
    store.close()

    # --- Display results for ALL models ---
    models_in_scores = sorted({s.model for s in all_scores})
    click.echo(f"\n--- Results ({len(models_in_scores)} models, {len(all_scores)} scores) ---")

    for m in models_in_scores:
        m_scores = [s for s in all_scores if s.model == m]
        avg_ig = sum(s.ig_normalized for s in m_scores) / len(m_scores)
        avg_is = sum(s.is_normalized for s in m_scores) / len(m_scores)
        avg_hm = sum(s.harmonic_mean for s in m_scores) / len(m_scores)
        marker = " <-- current" if m == model else ""
        click.echo(
            f"  {m}: IG_F1={avg_ig:.3f}  IS={avg_is:.3f}  "
            f"HM={avg_hm:.3f}  ({len(m_scores)} users){marker}"
        )

    # Per-user detail for current model
    current_scores = [s for s in all_scores if s.model == model]
    if current_scores:
        click.echo(f"\n--- Per-user detail: {model} ---")
        for s in current_scores:
            click.echo(
                f"  User {s.user_id}: "
                f"IG_F1={s.ig_normalized:.3f} "
                f"IS={s.is_normalized:.3f} "
                f"HM={s.harmonic_mean:.3f} "
                f"(oracle={s.oracle_count})"
            )

    # Save results
    if output:
        results = [
            {
                "user_id": s.user_id,
                "dataset": s.dataset,
                "model": s.model,
                "ig_normalized": s.ig_normalized,
                "is_normalized": s.is_normalized,
                "harmonic_mean": s.harmonic_mean,
                "oracle_count": s.oracle_count,
            }
            for s in all_scores
        ]
        Path(output).write_text(json.dumps(results, indent=2))
        click.echo(f"\nResults saved to {output}")

    report_path = _resolve_report_path(report, report_dir, dataset_name)
    if report_path:
        md = _generate_report(all_scores, dataset_name)
        report_path.write_text(md)
        click.echo(f"Report saved to {report_path}")


@main.command("export-oracle")
@click.option("--results-db", required=True, help="Path to the SQLite results DB to export from.")
@click.option("--dataset", "-d", required=True, help="Dataset name to export.")
@click.option("--output", "-o", required=True, help="Output JSON path (e.g., oracle_synthetic.json).")
@click.option("--merge", "merge_path", default=None, help="Existing oracle JSON to union with (e.g., gistbench/assets/oracle_synthetic.json).")
def export_oracle(results_db: str, dataset: str, output: str, merge_path: str | None) -> None:
    """Snapshot the cross-model oracle from a results DB to a JSON file.

    For each user the oracle is the union of verified+mapped interest
    categories across every model in the store. Pass --merge to union
    with an existing oracle file (e.g., to extend the bundled snapshot).
    """
    store = ResultsStore(results_db)
    try:
        store_oracle = Oracle.from_results_store(store, dataset)
    finally:
        store.close()

    click.echo(f"Store contributes {len(store_oracle.category_ids)} users for dataset '{dataset}'")

    if merge_path:
        existing = Oracle.from_file(merge_path)
        click.echo(f"Merging with {merge_path} ({len(existing.category_ids)} users)")
        oracle = existing.merge(store_oracle)
    else:
        oracle = store_oracle

    if not oracle.category_ids:
        click.echo(
            f"Warning: no oracle entries to write. Has any model run with taxonomy "
            f"mapping enabled for dataset '{dataset}'?",
            err=True,
        )

    oracle.to_file(output, dataset=dataset)
    click.echo(f"Wrote {len(oracle.category_ids)} users to {output}")


@main.command()
def datasets() -> None:
    """List available datasets with built-in configurations."""
    click.echo("Available datasets:")
    for name, cfg in DATASET_CONFIGS.items():
        signals = []
        if cfg.has_explicit_positive:
            signals.append("explicit+")
        if cfg.has_implicit_positive:
            signals.append("implicit+")
        if cfg.has_implicit_negative:
            signals.append("implicit-")
        if cfg.has_explicit_negative:
            signals.append("explicit-")
        click.echo(f"  {name:25s} {cfg.object_name:15s} [{', '.join(signals)}]")


@main.command()
@click.option("--dataset", "-d", default="synthetic", help="Dataset name to download.")
@click.option("--cache-dir", default=None, help="Cache directory.")
def download(dataset: str, cache_dir: str | None) -> None:
    """Download a dataset from Hugging Face (or load mock data)."""
    df = download_dataset(dataset, cache_dir=cache_dir)
    click.echo(
        f"Loaded {len(df)} engagements for {df['user_id'].nunique()} users."
    )


SMOKE_TEST_MODELS = ["gpt-4o-mini"]
SMOKE_TEST_DATASETS = ["mock", "synthetic"]


@main.command("smoke-test")
@click.option("--base-url", default=None, help="OpenAI-compatible API base URL (e.g., http://localhost:11434/v1 for Ollama).")
@click.option("--api-key", default=None, help="API key (defaults to OPENAI_API_KEY env var; auto-set to a placeholder when --base-url is provided).")
@click.option("--num-users", "-n", default=5, show_default=True, help="Number of users to evaluate.")
@click.option("--models", default=",".join(SMOKE_TEST_MODELS), show_default=True, help="Comma-separated list of models to evaluate.")
@click.option("--datasets", "datasets_arg", default=",".join(SMOKE_TEST_DATASETS), show_default=True, help=f"Comma-separated dataset cases to run. Choices: {', '.join(SMOKE_TEST_DATASETS)}.")
@click.option("--report", default=None, help="Explicit path to write a markdown report (overrides --report-dir).")
@click.option("--report-dir", default=None, help="Directory for timestamped markdown reports (e.g., 'reports'). Created if missing.")
def smoke_test(
    base_url: str | None,
    api_key: str | None,
    num_users: int,
    models: str,
    datasets_arg: str,
    report: str | None,
    report_dir: str | None,
) -> None:
    """Run a quick e2e validation against mock and/or synthetic datasets.

    Evaluates N users across one or more models for the selected datasets.
    Both cases use bundled oracles, so a single model is sufficient.

    Works with any OpenAI-compatible endpoint via --base-url (Ollama, vLLM,
    LM Studio, etc.). When --base-url is set, --api-key defaults to a
    placeholder so local servers that don't authenticate just work.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and base_url:
        api_key = "local"  # local OpenAI-compatible servers ignore the key
    if not api_key:
        click.echo("Error: No API key provided. Set OPENAI_API_KEY or use --api-key.", err=True)
        sys.exit(1)

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        click.echo("Error: --models is empty.", err=True)
        sys.exit(1)

    dataset_list = [d.strip() for d in datasets_arg.split(",") if d.strip()]
    unknown = [d for d in dataset_list if d not in SMOKE_TEST_DATASETS]
    if unknown:
        click.echo(
            f"Error: unknown dataset(s) {unknown}. Choices: {SMOKE_TEST_DATASETS}.",
            err=True,
        )
        sys.exit(1)
    if not dataset_list:
        click.echo("Error: --datasets is empty.", err=True)
        sys.exit(1)

    passed = 0
    failed = 0
    report_sections: list[str] = []

    for label, df, dataset_name, oracle in _smoke_test_cases(num_users, dataset_list):
        click.echo(f"\n{'=' * 60}")
        click.echo(f"SMOKE TEST: {label}")
        click.echo(f"  {len(df)} engagements, {df['user_id'].nunique()} users, "
                    f"evaluating {num_users}")
        click.echo(f"  Models: {', '.join(model_list)}")
        click.echo(f"{'=' * 60}")

        validate_dataset(df)
        user_ids = sample_users(df, n=num_users, seed=42)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            store = ResultsStore(db_path)

            for model_name in model_list:
                click.echo(f"\n--- {model_name} ---")
                client = OpenAIClient(model=model_name, api_key=api_key, base_url=base_url)
                run_benchmark(
                    client=client,
                    df=df,
                    user_ids=user_ids,
                    oracle=None,
                    dataset_name=dataset_name,
                    model_name=model_name,
                    results_store=store,
                )

            # Merge bundled oracle with store contributions (mirrors `run`).
            store_oracle = Oracle.from_results_store(store, dataset_name)
            merged_oracle = oracle.merge(store_oracle)
            all_scores = store.rescore_with_oracle(dataset_name, merged_oracle)

            store.close()

            # Validate results
            models_scored = sorted({s.model for s in all_scores})
            assert len(models_scored) == len(model_list), (
                f"Expected {len(model_list)} models scored, got {len(models_scored)}"
            )
            for s in all_scores:
                assert 0.0 <= s.ig_normalized <= 1.0, f"IG out of range: {s.ig_normalized}"
                assert 0.0 <= s.is_normalized <= 1.0, f"IS out of range: {s.is_normalized}"
                assert 0.0 <= s.harmonic_mean <= 1.0, f"HM out of range: {s.harmonic_mean}"

            click.echo(f"\n--- Results ---")
            for m in models_scored:
                m_scores = [s for s in all_scores if s.model == m]
                avg_ig = sum(s.ig_normalized for s in m_scores) / len(m_scores)
                avg_is = sum(s.is_normalized for s in m_scores) / len(m_scores)
                avg_hm = sum(s.harmonic_mean for s in m_scores) / len(m_scores)
                click.echo(
                    f"  {m}: IG_F1={avg_ig:.3f}  IS={avg_is:.3f}  "
                    f"HM={avg_hm:.3f}  ({len(m_scores)} users)"
                )

            click.echo(f"\nPASSED: {label}")
            passed += 1

            report_sections.append(
                _generate_report(all_scores, dataset_name, title=f"Smoke Test: {label}")
            )

        except Exception as e:
            click.echo(f"\nFAILED: {label} — {e}", err=True)
            failed += 1
            report_sections.append(f"# Smoke Test: {label}\n\n**FAILED:** {e}\n")
        finally:
            Path(db_path).unlink(missing_ok=True)

    click.echo(f"\n{'=' * 60}")
    click.echo(f"SMOKE TEST SUMMARY: {passed} passed, {failed} failed")
    click.echo(f"{'=' * 60}")

    report_path = _resolve_report_path(report, report_dir, "smoke-test")
    if report_path:
        header = (
            f"# GISTBench Smoke Test Report\n\n"
            f"**Date:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  \n"
            f"**Models:** {', '.join(model_list)}  \n"
            f"**Users per test:** {num_users}  \n"
            f"**Result:** {passed} passed, {failed} failed\n\n---\n\n"
        )
        report_path.write_text(header + "\n---\n\n".join(report_sections))
        click.echo(f"Report saved to {report_path}")

    sys.exit(1 if failed else 0)


def _smoke_test_cases(num_users: int, dataset_names: list[str]):
    """Yield (label, df, dataset_name, oracle) for each selected case."""
    if "mock" in dataset_names:
        click.echo("Loading mock dataset...")
        mock_df = load_mock_dataset()
        mock_oracle = load_mock_oracle()
        yield ("Mock dataset (bundled oracle)", mock_df, "mock", mock_oracle)

    if "synthetic" in dataset_names:
        click.echo("Downloading synthetic dataset...")
        real_df = download_dataset("synthetic")
        real_oracle = load_bundled_oracle()
        yield (
            f"Synthetic dataset (N={num_users}, bundled oracle)",
            real_df,
            "synthetic",
            real_oracle,
        )
