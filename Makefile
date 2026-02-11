PYTHON ?= python3

.PHONY: setup lint format test train report

setup:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	ruff check .
	black --check .
	isort --check-only .

format:
	ruff check . --fix
	black .
	isort .

test:
	pytest -q

train:
	$(PYTHON) -m src.models.train --config configs/base.yaml

report:
	$(PYTHON) scripts/generate_executive_summary.py
