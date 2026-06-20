.PHONY: install data local-export ingest paper-session backtest paper test clean all dashboard

install:
	pip install -e .

data:
	python -m quant --generate-example-data

local-export:
	python -m quant --write-synthetic-local-export data/raw/resset_daily.csv --column-mapping configs/data_mappings/resset_daily_illustrative.yaml --symbols 000001 000002 000003 --start 2020-01-01 --end 2020-01-31

ingest:
	python -m quant --ingest-local-data data/raw/resset_daily.csv --column-mapping configs/data_mappings/resset_daily_illustrative.yaml --symbols 000001 000002 000003 --adjustment-convention backward --adjusted-price-column

backtest:
	python -m quant --run-config configs/experiments/exp_placeholder.yaml

paper:
	python -m quant --paper-demo

paper-session:
	python scripts/run_paper_session.py --data data/processed/local_daily_ohlcv.parquet --symbols 000001 000002 000003

manual-quote-step:
	python scripts/run_paper_session.py --manual-quotes data/raw/manual_quotes.csv --symbols 000001 000002 000003 --as-of 2020-01-31

test:
	python -m unittest discover -s tests

dashboard:
	streamlit run dashboard/app_streamlit.py

all: data backtest paper test

clean:
	rm -rf results/* logs/* data/processed/* data/raw/*
	find . -type d -name __pycache__ -exec rm -rf {} +
