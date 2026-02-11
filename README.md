# CharityML Donor Prediction - Remote-Ready Refactor

This repository now includes a reproducible ML pipeline and engineering scaffolding aligned with modern remote data analyst / applied AI expectations.

## What changed
- Moved core logic into `src/` modules for loading, validation, feature prep, training, evaluation, inference, and drift monitoring.
- Added config-driven training (`configs/base.yaml`).
- Added testing (`tests/`) and quality tooling (`ruff`, `black`, `isort`, `mypy`).
- Added CI workflow for lint + tests.
- Added `Makefile` commands for standard local workflows.

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
make train
make test
```

## Main commands
- `make train` - train model pipeline and write artifacts/reports.
- `make test` - run tests.
- `make lint` - run style/lint checks.
- `make report` - generate executive summary.

## Outputs
- `artifacts/charity_donor_pipeline.joblib` - trained pipeline + threshold.
- `artifacts/model_features.txt` - transformed model features.
- `reports/model_card.md` - model validation summary.
- `reports/executive_summary.md` - stakeholder summary.

## Notes on external integrations
This repo is ready for local execution. External platform integrations (MLflow server, cloud scheduler, BI tool publishing, production API hosting) need credentials and environment details from you.
