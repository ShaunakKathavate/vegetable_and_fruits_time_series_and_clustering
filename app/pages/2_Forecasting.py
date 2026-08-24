import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import compute_all_forecasts, load_forecast_metadata, load_weekly_panel
from data_pipeline import COMMODITY_COL

st.set_page_config(page_title="Forecasting", layout="wide")
st.title("Price Forecasting")

meta = load_forecast_metadata()
st.caption(
    f"Model: a Temporal Fusion Transformer (`{meta['final_model']}`), trained once across "
    "every commodity's weekly history (notebooks/05_forecasting_backtest.ipynb, section 11), "
    "loaded from models/forecast_final_model/. It won a rolling-origin backtest against "
    "SARIMA, Prophet, LSTM and two baselines -- see the comparison below."
)

with st.expander("Backtest comparison (why TFT was chosen)"):
    combined = {**meta["candidate_backtest_mape"], **meta["baseline_backtest_mape"]}
    bt = pd.DataFrame(list(combined.items()), columns=["model", "avg_mape_pct"])
    bt = bt.sort_values("avg_mape_pct")
    st.dataframe(bt, use_container_width=True, hide_index=True)
    st.caption(f"Source: {meta['backtest_cache_source']}")

weekly = load_weekly_panel()
forecast = compute_all_forecasts()

commodities = sorted(forecast["unique_id"].unique().tolist())
default_idx = commodities.index(meta["target_commodity"]) if meta["target_commodity"] in commodities else 0
selected = st.selectbox("Select a commodity", commodities, index=default_idx)

hist = (weekly[weekly[COMMODITY_COL] == selected]
        .set_index("Date")["avg_price"].sort_index().tail(52))
fc = forecast[forecast["unique_id"] == selected].sort_values("ds")

fig = go.Figure()
fig.add_scatter(x=hist.index, y=hist.values, mode="lines", name="Historical (last 52 weeks)",
                 line=dict(color="black"))
if len(hist) and len(fc):
    fig.add_scatter(x=[hist.index[-1], fc["ds"].iloc[0]], y=[hist.iloc[-1], fc["forecast"].iloc[0]],
                     mode="lines", line=dict(color="royalblue", dash="dash"), opacity=0.4,
                     showlegend=False)
    fig.add_vline(x=hist.index[-1], line=dict(color="gray", dash="dash"))
fig.add_scatter(x=fc["ds"], y=fc["forecast"], mode="lines+markers",
                 name=f"Forecast ({meta['final_model']})", line=dict(color="royalblue"))
fig.update_layout(title=f"{selected} — {len(fc)}-Week Forecast", yaxis_title="Average price",
                   height=500)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Forecast values")
st.dataframe(fc[["ds", "forecast"]].rename(columns={"ds": "Date", "forecast": "Forecast"}),
             use_container_width=True, hide_index=True)
