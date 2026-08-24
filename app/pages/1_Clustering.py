import pandas as pd
import plotly.express as px
import streamlit as st

from utils import compute_cluster_table

st.set_page_config(page_title="Clustering", layout="wide")
st.title("Commodity Clustering")
st.caption(
    "Cluster labels come from the saved models/volatility_cluster_pipeline.joblib and "
    "models/seasonality_cluster_pipeline.joblib (StandardScaler + KMeans), fit in "
    "notebooks/03_commodity_clustering.ipynb. Shape clusters (seasonal-curve DTW) come from "
    "that notebook's own cached output, since DTW clustering has no single fitted model to reuse."
)


def fmt(row, col, template="{:.2f}"):
    value = row.get(col)
    if value is None or pd.isna(value):
        return "n/a (short history)"
    return template.format(value)


table, meta = compute_cluster_table()

commodities = sorted(table.index.tolist())
selected = st.selectbox("Select a commodity", commodities,
                         index=commodities.index("Tomato Big(Nepali)") if "Tomato Big(Nepali)" in commodities else 0)

row = table.loc[selected]
col1, col2, col3 = st.columns(3)
col1.metric("Volatility cluster", row["volatility_cluster_name"])
col2.metric("Volatility + seasonality cluster", fmt(row, "seasonality_cluster_name", "{}"))
col3.metric("Seasonal shape cluster", fmt(row, "shape_cluster_name", "{}"))

stat1, stat2, stat3, stat4 = st.columns(4)
stat1.metric("Coefficient of variation", fmt(row, "cv"))
stat2.metric("Avg daily min-max range", fmt(row, "avg_daily_range_pct", "{:.1%}"))
stat3.metric("Seasonal strength", fmt(row, "seasonal_strength"))
stat4.metric("Mean price", fmt(row, "price_mean", "{:.1f}"))

st.divider()
st.subheader("Volatility landscape")
plot_df = table.reset_index().rename(columns={"index": "Commodity"})
plot_df["is_selected"] = plot_df["Commodity"] == selected
fig = px.scatter(
    plot_df, x="cv", y="avg_daily_range_pct", color="volatility_cluster_name",
    hover_name="Commodity",
    labels={"cv": "Coefficient of variation (price)", "avg_daily_range_pct": "Avg daily min-max range %",
            "volatility_cluster_name": "Cluster"},
    title="Commodities by price volatility",
)
sel_row = plot_df[plot_df["is_selected"]]
fig.add_scatter(x=sel_row["cv"], y=sel_row["avg_daily_range_pct"], mode="markers",
                 marker=dict(size=18, symbol="star", color="black"),
                 name=f"{selected} (selected)")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("All commodities")
display_cols = [c for c in ["volatility_cluster_name", "seasonality_cluster_name", "shape_cluster_name",
                             "cv", "avg_daily_range_pct", "seasonal_strength", "price_mean", "tier"]
                 if c in table.columns]
st.dataframe(table[display_cols].sort_values("cv", ascending=False), use_container_width=True)
