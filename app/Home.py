import streamlit as st

from utils import load_raw_data, load_forecast_metadata
from data_pipeline import DATE_COL, COMMODITY_COL

st.set_page_config(page_title="Kalimati Commodity Intelligence", layout="wide")

st.title("Kalimati Commodity Price Intelligence")
st.markdown(
    """
This app turns the analysis in `notebooks/` into an interactive tool, built on three pieces
of work:

- **Clustering** (`03_commodity_clustering.ipynb`) — groups commodities by price behavior
  (volatility, seasonality, seasonal shape), using the saved `models/*_cluster_pipeline.joblib`
  models.
- **Forecasting** (`05_forecasting_backtest.ipynb`) — a backtested Temporal Fusion Transformer
  (the model that won a rolling-origin backtest against SARIMA, Prophet, LSTM, and two
  baselines), saved to `models/forecast_final_model/`.
- **Business recommendations** (`04_business_simulations_clusters.ipynb` and
  `06_advanced_business_simulations.ipynb`) — Monte Carlo price simulations (STL decomposition
  + block-bootstrapped residuals) turned into decision-support tools: best month to sell,
  whether today's price is a good deal, portfolio diversification across clusters, and
  buy-now-vs-wait-and-see.

Use the sidebar to navigate between pages.
"""
)

df = load_raw_data()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Commodities", f"{df[COMMODITY_COL].nunique()}")
col2.metric("Price observations", f"{len(df):,}")
col3.metric("Date range start", str(df[DATE_COL].min().date()))
col4.metric("Date range end", str(df[DATE_COL].max().date()))

st.divider()
st.subheader("Forecasting model summary")
meta = load_forecast_metadata()
mcol1, mcol2, mcol3 = st.columns(3)
mcol1.metric("Winning model", meta["final_model"])
mcol2.metric("Backtest MAPE", f"{meta['candidate_backtest_mape'][meta['final_model']]:.1f}%")
best_baseline = min(meta["baseline_backtest_mape"], key=meta["baseline_backtest_mape"].get)
mcol3.metric(f"Beats baseline ({best_baseline})",
             "Yes" if meta["beats_strongest_baseline"] else "No")
st.caption(
    "Backtest MAPE compares the deployable candidates {SARIMA, Prophet, LSTM, TFT} averaged "
    "across a sample of commodities via rolling-origin backtesting -- see "
    "reports/from notebooks/forecasting_outputs/backtest_all_commodities.csv."
)
