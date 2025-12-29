# Repository Guidelines

## Project Structure & Module Organization
- `torchmdnet/` contains the core Python package (models, datasets, training entry points).
- `scripts/` provides lightweight wrappers; `torchmdnet/scripts/train.py` backs the `torchmd-train` CLI.
- `tests/` holds the pytest suite; helpers live in `tests/utils.py`.
- `examples/` includes runnable training configs (for example `examples/ET-QM9.yaml`) and small demo scripts.
- `docs/`, `benchmarks/`, and `cibuildwheel_support/` support documentation, performance checks, and packaging.

## Build, Test, and Development Commands
- `pip install -e .` installs the package in editable mode for local development.
- `torchmd-train --conf examples/ET-QM9.yaml --log-dir output/` runs a training job with an example config.
- `torchmd-train --help` lists all CLI options and defaults.
- `pytest` runs the full test suite from the repository root.
- `make release version=x.y.z` and `make docs version=x.y.z` create release/docs tags (maintainers only).

## Coding Style & Naming Conventions
- Format Python with `black` before committing (project uses Black).
- Use 4-space indentation; prefer `snake_case` for functions/variables and `CapWords` for classes.
- Keep module/file names lowercase with underscores, matching existing package layout.

## Testing Guidelines
- Tests live in `tests/` and follow the `test_*.py` naming pattern.
- Add or update tests alongside new features or model changes; there is no stated coverage target.
- If you introduce new configs or behaviors, add a focused regression test and update `tests/utils.py` helpers as needed.

## Commit & Pull Request Guidelines
- Recent history uses short, descriptive commit messages (often lowercase, no prefixes); keep them concise.
- PRs should include a clear summary, rationale, and the tests you ran (or why not).
- Link related issues and call out any config, dataset, or training behavior changes.
