.PHONY: setup test lint clean

setup:
	pip install -r requirements.txt
	nbstripout --install

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f data/census.db
