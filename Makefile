.PHONY: install data backtest paper test clean all dashboard

install:
	pip install -e .[dev]

data:
	python scripts/make_synthetic_data.py

backtest:
	python scripts/run_backtest.py --config configs/experiments/exp_placeholder.yaml

paper:
	python scripts/run_paper_demo.py

test:
	pytest

dashboard:
	streamlit run dashboard/app_streamlit.py

all: data backtest paper test

clean:
	rm -rf results/* logs/* data/processed/* data/raw/*
	find . -type d -name __pycache__ -exec rm -rf {} +
