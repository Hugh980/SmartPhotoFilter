.PHONY: install install-dev test lint format run run-cli benchmark download-models export-models

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,gui,ml,train]"

test:
	pytest

lint:
	ruff check src tests

format:
	ruff format src tests

run:
	python -m src.main --gui

run-cli:
	python -m src.main --input ./input --output ./output

benchmark:
	python scripts/benchmark.py

download-models:
	python scripts/download_models.py

export-models:
	python scripts/export_coreml.py
