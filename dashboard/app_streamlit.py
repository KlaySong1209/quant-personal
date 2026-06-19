"""Local Streamlit dashboard for quant-personal.

This is a view layer only. It imports orchestration from ``quant.app`` and does
not implement data, strategy, or backtest logic.
"""

from __future__ import annotations

from pathlib import Path

from quant import app

STREAMLIT_INSTALL_COMMAND = app.STREAMLIT_INSTALL_COMMAND

try:  # pragma: no cover - exercised by import smoke test when unavailable
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover
    st = None


def _require_streamlit():
    if st is None:
        print(f"Streamlit is not installed. Install it with: {STREAMLIT_INSTALL_COMMAND}")
        return False
    return True


def _metrics_table(metrics: dict[str, float]):
    explanations = app.metric_explanations()
    rows = [
        {"metric": k, "value": v, "meaning": explanations.get(k, k)}
        for k, v in metrics.items()
    ]
    st.dataframe(rows, use_container_width=True)


def _data_page():
    st.header("Data")
    source = st.radio("Data source", ["synthetic", "local CSV config"], horizontal=True)
    if source == "synthetic":
        if st.button("Generate example data"):
            paths = app.generate_example_data()
            st.success(f"Wrote {len(paths)} files to data/example/")
            st.write([str(p) for p in paths])
    else:
        st.info("Use configs/default.yaml data.csv_path and column mappings for local exports.")
    configs = app.list_experiment_configs()
    st.write("Available configs:", [p.name for p in configs])


def _run_page():
    st.header("Configure & Run")
    configs = app.list_experiment_configs()
    if not configs:
        st.warning("No configs found in configs/experiments/.")
        return
    selected = st.selectbox("Experiment config", configs, format_func=lambda p: p.name)
    st.caption("Edit key parameters in the YAML config for now; this panel runs the shared src/quant path.")
    if st.button("Run backtest"):
        artifacts = app.run_backtest_experiment(Path(selected))
        st.success(f"Run complete: {artifacts.run_id}")
        st.code(str(artifacts.run_dir))
        _metrics_table(artifacts.metrics)


def _results_page():
    st.header("Results")
    runs = app.list_runs()
    if not runs:
        st.info("No runs yet.")
        return
    run = st.selectbox("Run", runs, format_func=lambda p: p.name)
    summary = app.load_run_summary(run)
    _metrics_table(summary["metrics"])
    eq = app.load_equity_curve(run)
    st.line_chart(eq)
    trades = app.load_trades(run)
    st.dataframe(trades, use_container_width=True)
    metadata = summary.get("metadata", {})
    st.subheader("Data honesty facts")
    st.json(
        {
            "git_commit": metadata.get("git_commit"),
            "git_dirty": metadata.get("git_dirty"),
            "input_data_hashes": metadata.get("input_data_hashes", {}),
        }
    )


def _experiments_page():
    st.header("Experiments")
    runs = app.list_runs()
    if len(runs) < 2:
        st.info("Need at least two runs to compare.")
        return
    left = st.selectbox("Left run", runs, format_func=lambda p: p.name, key="left")
    right = st.selectbox("Right run", runs, format_func=lambda p: p.name, key="right")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(left.name)
        _metrics_table(app.load_run_summary(left)["metrics"])
    with col2:
        st.subheader(right.name)
        _metrics_table(app.load_run_summary(right)["metrics"])


def main() -> None:
    if not _require_streamlit():
        return
    st.set_page_config(page_title="quant-personal", layout="wide")
    st.title("quant-personal")
    page = st.sidebar.radio("Page", ["Data", "Configure & run", "Results", "Experiments"])
    if page == "Data":
        _data_page()
    elif page == "Configure & run":
        _run_page()
    elif page == "Results":
        _results_page()
    else:
        _experiments_page()


if __name__ == "__main__":  # pragma: no cover
    main()

