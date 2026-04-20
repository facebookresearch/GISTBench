"""SQLite-backed results store for GISTBench.

Persists per-interest IG/IS results, interest-to-category mappings, and
enables cross-model oracle computation.  This is essential for correct
IG Recall: the oracle count must reflect verified categories across ALL
evaluated models, not just the current one.

Usage::

    store = ResultsStore("gistbench_results.db")

    # After evaluating each model, save results
    store.save_ig_results(ig_results)
    store.save_is_results(is_results)
    store.save_category_map(category_map)

    # Compute oracle and re-score with correct cross-model denominator
    oracle = store.compute_oracle(user_id, dataset)
    score = compute_user_score(..., oracle_count=oracle)

    # Or use the convenience method
    scores = store.rescore_all(dataset)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from gistbench.schema import IGResult, ISResult, UserScore
from gistbench.steps.scoring import compute_user_score


class ResultsStore:
    """SQLite store for IG/IS results and taxonomy mappings."""

    def __init__(self, db_path: str | Path = "gistbench_results.db") -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._create_tables()

    def _create_tables(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS ig_results (
                model TEXT NOT NULL,
                dataset TEXT NOT NULL,
                user_id TEXT NOT NULL,
                interest TEXT NOT NULL,
                verified INTEGER NOT NULL,
                PRIMARY KEY (model, dataset, user_id, interest)
            );

            CREATE TABLE IF NOT EXISTS is_results (
                model TEXT NOT NULL,
                dataset TEXT NOT NULL,
                user_id TEXT NOT NULL,
                interest TEXT NOT NULL,
                correct INTEGER NOT NULL,
                selected INTEGER NOT NULL,
                backing INTEGER NOT NULL,
                PRIMARY KEY (model, dataset, user_id, interest)
            );

            CREATE TABLE IF NOT EXISTS taxonomy_map (
                interest TEXT PRIMARY KEY,
                category_id INTEGER NOT NULL
            );
        """)
        self._conn.commit()

    # --- IG results ---

    def save_ig_results(self, results: list[IGResult]) -> None:
        """Save IG verification results (upsert)."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO ig_results "
            "(model, dataset, user_id, interest, verified) "
            "VALUES (?, ?, ?, ?, ?)",
            [(r.model, r.dataset, r.user_id, r.interest, int(r.verified))
             for r in results],
        )
        self._conn.commit()

    def get_ig_results(
        self, model: str, dataset: str, user_id: str
    ) -> list[IGResult]:
        """Load IG results for a specific (model, dataset, user)."""
        rows = self._conn.execute(
            "SELECT model, dataset, user_id, interest, verified "
            "FROM ig_results WHERE model = ? AND dataset = ? AND user_id = ?",
            (model, dataset, user_id),
        ).fetchall()
        return [
            IGResult(
                model=r[0], dataset=r[1], user_id=r[2],
                interest=r[3], verified=bool(r[4]),
            )
            for r in rows
        ]

    # --- IS results ---

    def save_is_results(self, results: list[ISResult]) -> None:
        """Save IS specificity results (upsert)."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO is_results "
            "(model, dataset, user_id, interest, correct, selected, backing) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(r.model, r.dataset, r.user_id, r.interest,
              r.correct, r.selected, r.backing)
             for r in results],
        )
        self._conn.commit()

    def get_is_results(
        self, model: str, dataset: str, user_id: str
    ) -> list[ISResult]:
        """Load IS results for a specific (model, dataset, user)."""
        rows = self._conn.execute(
            "SELECT model, dataset, user_id, interest, correct, selected, backing "
            "FROM is_results WHERE model = ? AND dataset = ? AND user_id = ?",
            (model, dataset, user_id),
        ).fetchall()
        return [
            ISResult(
                model=r[0], dataset=r[1], user_id=r[2],
                interest=r[3], correct=r[4], selected=r[5], backing=r[6],
            )
            for r in rows
        ]

    # --- Taxonomy mappings ---

    def save_category_map(self, mapping: dict[str, int]) -> None:
        """Save interest → category_id mappings (upsert)."""
        self._conn.executemany(
            "INSERT OR REPLACE INTO taxonomy_map (interest, category_id) "
            "VALUES (?, ?)",
            list(mapping.items()),
        )
        self._conn.commit()

    def get_category_map(self) -> dict[str, int]:
        """Load all interest → category_id mappings."""
        rows = self._conn.execute(
            "SELECT interest, category_id FROM taxonomy_map"
        ).fetchall()
        return dict(rows)

    def get_category(self, interest: str) -> int | None:
        """Look up a single interest's category_id."""
        row = self._conn.execute(
            "SELECT category_id FROM taxonomy_map WHERE interest = ?",
            (interest,),
        ).fetchone()
        return row[0] if row else None

    # --- Oracle computation ---

    def compute_oracle(self, user_id: str, dataset: str) -> int:
        """Compute the oracle count for a user across all models.

        Oracle = number of distinct taxonomy categories with ≥1 verified
        interest across ANY model.  This is the cross-model denominator
        for IG Recall.

        Interests without a taxonomy mapping are dropped — only mapped
        interests contribute to the oracle.
        """
        return len(self.compute_oracle_categories(user_id, dataset))

    def compute_oracle_categories(self, user_id: str, dataset: str) -> list[int]:
        """Return the distinct verified category IDs for a user across all models.

        This is the detailed version of ``compute_oracle`` — returns the actual
        category IDs rather than just the count.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT tm.category_id "
            "FROM ig_results ig "
            "JOIN taxonomy_map tm ON ig.interest = tm.interest "
            "WHERE ig.user_id = ? AND ig.dataset = ? AND ig.verified = 1",
            (user_id, dataset),
        ).fetchall()
        return sorted(r[0] for r in rows)

    # --- Convenience: list models/users ---

    def list_models(self, dataset: str | None = None) -> list[str]:
        """List all models with stored results."""
        if dataset:
            rows = self._conn.execute(
                "SELECT DISTINCT model FROM ig_results WHERE dataset = ? ORDER BY model",
                (dataset,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT DISTINCT model FROM ig_results ORDER BY model"
            ).fetchall()
        return [r[0] for r in rows]

    def list_users(self, dataset: str, model: str | None = None) -> list[str]:
        """List all users with stored results."""
        if model:
            rows = self._conn.execute(
                "SELECT DISTINCT user_id FROM ig_results "
                "WHERE dataset = ? AND model = ? ORDER BY user_id",
                (dataset, model),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT DISTINCT user_id FROM ig_results "
                "WHERE dataset = ? ORDER BY user_id",
                (dataset,),
            ).fetchall()
        return [r[0] for r in rows]

    # --- Re-scoring with cross-model oracle ---

    def models_for_user(self, user_id: str, dataset: str) -> list[str]:
        """List models that have evaluated a specific user."""
        rows = self._conn.execute(
            "SELECT DISTINCT model FROM ig_results "
            "WHERE user_id = ? AND dataset = ? ORDER BY model",
            (user_id, dataset),
        ).fetchall()
        return [r[0] for r in rows]

    def rescore_all(
        self, dataset: str, min_models: int = 3,
    ) -> list[UserScore]:
        """Re-compute scores for all (model, user) pairs using cross-model oracle.

        This is the correct way to compute final scores: run all models first,
        save their results, then call this to get scores with the proper
        cross-model oracle denominator for IG Recall.

        Args:
            dataset: Dataset name.
            min_models: Minimum number of models that must have evaluated a
                user before that user can be scored.  The oracle is only
                meaningful when computed across multiple models.

        Raises:
            ValueError: If fewer than ``min_models`` models have stored
                results for the dataset.
        """
        models = self.list_models(dataset)
        if len(models) < min_models:
            raise ValueError(
                f"Cannot score: only {len(models)} model(s) in the store "
                f"for dataset '{dataset}', but min_models={min_models}. "
                f"Run at least {min_models} models first."
            )

        category_map = self.get_category_map() or None
        all_users = self.list_users(dataset)

        scores = []
        for user_id in all_users:
            # Only score users evaluated by enough models
            user_models = self.models_for_user(user_id, dataset)
            if len(user_models) < min_models:
                continue

            oracle = self.compute_oracle(user_id, dataset)
            for model in models:
                ig = self.get_ig_results(model, dataset, user_id)
                is_ = self.get_is_results(model, dataset, user_id)
                if not ig:
                    continue
                score = compute_user_score(
                    ig, is_, user_id, dataset, model,
                    oracle_count=oracle,
                    category_map=category_map,
                )
                scores.append(score)
        return scores

    def rescore_with_oracle(
        self, dataset: str, oracle: "Oracle",
    ) -> list[UserScore]:
        """Re-compute scores for all (model, user) pairs using a pre-computed oracle.

        Unlike ``rescore_all``, this uses the provided oracle directly
        instead of building one from the store.  No minimum model count
        is required.
        """
        category_map = self.get_category_map() or None
        models = self.list_models(dataset)
        all_users = self.list_users(dataset)

        scores = []
        for user_id in all_users:
            oc = oracle.count(user_id)
            if oc == 0:
                continue
            for model in models:
                ig = self.get_ig_results(model, dataset, user_id)
                is_ = self.get_is_results(model, dataset, user_id)
                if not ig:
                    continue
                score = compute_user_score(
                    ig, is_, user_id, dataset, model,
                    oracle_count=oc,
                    category_map=category_map,
                )
                scores.append(score)
        return scores

    def summary(self, dataset: str) -> dict:
        """Summary statistics for stored results."""
        models = self.list_models(dataset)
        users = self.list_users(dataset)
        taxonomy_count = self._conn.execute(
            "SELECT COUNT(*) FROM taxonomy_map"
        ).fetchone()[0]
        ig_count = self._conn.execute(
            "SELECT COUNT(*) FROM ig_results WHERE dataset = ?",
            (dataset,),
        ).fetchone()[0]
        is_count = self._conn.execute(
            "SELECT COUNT(*) FROM is_results WHERE dataset = ?",
            (dataset,),
        ).fetchone()[0]
        return {
            "dataset": dataset,
            "models": models,
            "users": len(users),
            "ig_results": ig_count,
            "is_results": is_count,
            "taxonomy_mappings": taxonomy_count,
        }

    # --- Import / Export ---

    def import_metrics_csv(self, path: str | Path) -> int:
        """Import per-interest results from a metrics CSV.

        Expected columns: model, dataset, user_id, interest,
        ig_verified, is_correct, is_selected, is_backing

        Returns:
            Number of rows imported.
        """
        df = pd.read_csv(path, sep=None, engine="python")  # auto-detect delimiter

        ig_results = []
        is_results = []
        for _, row in df.iterrows():
            ig_results.append(IGResult(
                model=str(row["model"]),
                dataset=str(row["dataset"]),
                user_id=str(row["user_id"]),
                interest=str(row["interest"]),
                verified=bool(int(row["ig_verified"])),
            ))
            is_results.append(ISResult(
                model=str(row["model"]),
                dataset=str(row["dataset"]),
                user_id=str(row["user_id"]),
                interest=str(row["interest"]),
                correct=int(row["is_correct"]),
                selected=int(row["is_selected"]),
                backing=int(row["is_backing"]),
            ))

        self.save_ig_results(ig_results)
        self.save_is_results(is_results)
        return len(ig_results)

    def import_taxonomy_csv(self, path: str | Path) -> int:
        """Import interest → category_id mappings from a taxonomy CSV.

        Expected columns: interest_text, category_id

        Returns:
            Number of mappings imported.
        """
        df = pd.read_csv(path, sep=None, engine="python")
        mapping: dict[str, int] = {}
        for _, row in df.iterrows():
            interest = str(row["interest_text"])
            cat_id = int(row["category_id"])
            if interest:
                mapping[interest] = cat_id

        self.save_category_map(mapping)
        return len(mapping)

    def export_metrics_csv(self, path: str | Path, dataset: str | None = None) -> int:
        """Export all stored results to a metrics CSV.

        Output columns: model, dataset, user_id, interest, ig_verified,
        is_correct, is_selected, is_backing

        Returns:
            Number of rows exported.
        """
        query = """
            SELECT ig.model, ig.dataset, ig.user_id, ig.interest,
                   ig.verified as ig_verified,
                   COALESCE(is_r.correct, 0) as is_correct,
                   COALESCE(is_r.selected, 0) as is_selected,
                   COALESCE(is_r.backing, 0) as is_backing
            FROM ig_results ig
            LEFT JOIN is_results is_r
                ON ig.model = is_r.model
                AND ig.dataset = is_r.dataset
                AND ig.user_id = is_r.user_id
                AND ig.interest = is_r.interest
        """
        params: tuple = ()
        if dataset:
            query += " WHERE ig.dataset = ?"
            params = (dataset,)
        query += " ORDER BY ig.model, ig.dataset, ig.user_id, ig.interest"

        df = pd.read_sql_query(query, self._conn, params=params)
        df.to_csv(path, index=False)
        return len(df)

    def export_taxonomy_csv(self, path: str | Path) -> int:
        """Export taxonomy mappings to CSV.

        Output columns: interest_text, category_id

        Returns:
            Number of mappings exported.
        """
        df = pd.read_sql_query(
            "SELECT interest as interest_text, category_id "
            "FROM taxonomy_map ORDER BY interest",
            self._conn,
        )
        df.to_csv(path, index=False)
        return len(df)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "ResultsStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
