.PHONY: install test lint train backtest paper dashboard clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

train:
	python -m src.main train

backtest:
	python -m src.main backtest

paper:
	python -m src.main paper

dashboard:
	python -m src.main dashboard

download:
	python -m src.main download

refresh:
	python -m src.main refresh

clean:
	rm -rf data/ models/ logs/ .venv/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
