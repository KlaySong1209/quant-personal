# quant-personal

Start here for non-engineers: see [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

One command:

```bash
python -m quant
```

Optional local dashboard:

```bash
pip install -e .[dashboard]
streamlit run dashboard/app_streamlit.py
```

This is a personal-use quantitative research system. It does not connect real
money, real brokers, credentials, live feeds, or validated alpha. Data is read
from local files or generated synthetically.

## Advanced CLI

```bash
python scripts/make_synthetic_data.py
python scripts/run_backtest.py --config configs/experiments/exp_placeholder.yaml
python scripts/run_paper_demo.py
pytest
```

Every run writes `results/<run_id>/` with config snapshot, metrics, equity curve,
trades, plots, logs, and metadata.
