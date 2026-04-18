# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] — 2026-04-07

### Added
- Initial public release of GISTBench evaluation framework.
- CLI (`gistbench run`, `gistbench datasets`, `gistbench download`) for running evaluations.
- Full pipeline: interest extraction, IG verification, IS evaluation, taxonomy mapping, and scoring.
- Cross-model oracle computation (auto-built from 3+ model runs).
- Pre-computed oracle support via `--oracle` flag.
- SQLite-backed results store with CSV import/export.
- Bundled 325-category interest taxonomy.
- Bundled oracle for the synthetic dataset (997 users).
- Six built-in dataset configurations: `synthetic`, `kuairec`, `mind`, `amazon_digital_music`, `yelp`, `goodreads`.
- Dataset download from Hugging Face (`facebook/gistbench`) via the `datasets` library.
- Mock dataset and oracle for offline testing.
- Integration test suite (no API key required).
- End-to-end test suite with real LLM inference.
- Support for any OpenAI-compatible API (OpenAI, Ollama, vLLM, etc.).

### Fixed
- Oracle deflation bug: `LEFT JOIN` → `JOIN` to drop unmapped interests from oracle count.
- Corrupt cache now falls back to re-download instead of silently failing.
