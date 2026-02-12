.PHONY: setup test lint clean run-app

setup:
	pip install -r requirements.txt
	nbstripout --install

test:
	python -m pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

run-app:
	streamlit run app/streamlit_app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -f data/census.db
