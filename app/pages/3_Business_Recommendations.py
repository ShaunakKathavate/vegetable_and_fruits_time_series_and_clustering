import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils import (
    compute_cluster_table,
    consumer_buy_timing_cached,
    diversification_sweep,
    eligible_simulation_commodities,
    load_raw_data,
    simulate_price_paths_cached,
)
import simulation_pipeline as sim
from data_pipeline import COMMODITY_COL, DATE_COL

st.set_page_config(page_title="Business Recommendations", layout="wide")
st.title("Business Recommendations")
st.caption(
    "Monte Carlo engine from notebooks/04_business_simulations_clusters.ipynb and "
    "notebooks/06_advanced_business_simulations.ipynb: STL-decompose each commodity's price "
    "into trend + seasonal + residual, extrapolate trend/seasonal forward, and block-bootstrap "
    "the historical residuals thousands of times to get a distribution of plausible future "
    "prices -- not just one point forecast."
)

commodities = eligible_simulation_commodities()
default_commodity = "Tomato Big(Nepali)" if "Tomato Big(Nepali)" in commodities else commodities[0]

tab1, tab2, tab3, tab4 = st.tabs([
    "Farmer: best month to sell",
    "Consumer: is today a good deal?",
    "Retailer: portfolio diversification",
    "Buyer: buy now vs. wait",
])

# ---------------------------------------------------------------------------
with tab1:
    st.subheader("When should a farmer plan to sell?")
    st.markdown(
        "Ranks future months by a **risk-adjusted score** (`expected_price - risk_aversion * "
        "std_price`), so the recommendation doesn't just chase the highest average price if "
        "that month is also wildly unreliable."
    )
    c1, c2, c3 = st.columns(3)
    commodity_1 = c1.selectbox("Commodity", commodities,
                                index=commodities.index(default_commodity), key="c1")
    risk_aversion = c2.slider("Risk aversion", 0.0, 2.0, sim.RISK_AVERSION, 0.1,
                               help="0 = purely maximize expected price; higher = favor lower volatility.")
    n_sims_1 = c3.select_slider("Simulations", [200, 500, 1000], value=500, key="n1")

    paths, hist, _ = simulate_price_paths_cached(commodity_1, horizon_days=365, n_sims=n_sims_1)
    window = sim.farmer_selling_window(paths, risk_aversion)
    best = window.iloc[0]
    highest_avg = window.sort_values("expected_price", ascending=False).iloc[0]

    st.success(
        f"**Recommended month: {int(best['month'])}** — expected price {best['expected_price']:.1f} "
        f"(typical range {best['p10']:.1f}-{best['p90']:.1f}, CV {best['cv']:.2f})"
    )
    if highest_avg["month"] != best["month"]:
        st.info(
            f"Month {int(highest_avg['month'])} has the highest raw average price "
            f"({highest_avg['expected_price']:.1f}) but scores lower once volatility is "
            f"factored in (CV {highest_avg['cv']:.2f} vs {best['cv']:.2f})."
        )

    order = window.sort_values("month")
    best_m = int(best["month"])
    colors = ["seagreen" if m == best_m else "royalblue" for m in order["month"]]
    fig = go.Figure()
    fig.add_bar(x=order["month"], y=order["expected_price"], marker_color=colors,
                error_y=dict(type="data", symmetric=False,
                              array=order["p90"] - order["expected_price"],
                              arrayminus=order["expected_price"] - order["p10"]))
    fig.update_layout(title=f"{commodity_1} — expected price by month (error bars = P10-P90)",
                       xaxis_title="Month", yaxis_title="Simulated price",
                       xaxis=dict(tickmode="linear", tick0=1, dtick=1), height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(window.round(2), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Is today's price a good deal?")
    st.markdown(
        "Compares a price against the simulated **seasonal-normal** distribution for that same "
        "calendar month -- not a raw historical average, so it accounts for both seasonality "
        "and the overall price trend."
    )
    df_raw = load_raw_data()
    c1, c2, c3 = st.columns(3)
    commodity_2 = c1.selectbox("Commodity", commodities,
                                index=commodities.index(default_commodity), key="c2")
    latest = df_raw[df_raw[COMMODITY_COL] == commodity_2].sort_values(DATE_COL).iloc[-1]
    current_price = c2.number_input("Price to evaluate", min_value=0.0,
                                     value=float(latest["Average"]), step=1.0)
    use_today = c3.checkbox("Use latest data date", value=True)
    current_date = latest[DATE_COL] if use_today else pd.Timestamp.today()

    result = consumer_buy_timing_cached(commodity_2, current_price, current_date,
                                         horizon_days=90, n_sims=500)

    verdict_color = {"Good time to buy": "success", "Expensive right now": "error",
                      "Roughly average pricing": "info"}[result["verdict"]]
    getattr(st, verdict_color)(result["message"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Typical median (that month)", result["typical_median"])
    m2.metric("Typical range (P10-P90)", f"{result['typical_p10']} - {result['typical_p90']}")
    m3.metric("Cheaper than", f"{result['cheaper_than_pct']:.0f}% of typical prices")

    fig = go.Figure()
    fig.add_histogram(x=result["typical_dist"], nbinsx=40,
                       name=f"Typical {result['month']} prices (simulated)",
                       marker_color="royalblue", opacity=0.65)
    fig.add_vline(x=result["current_price"], line=dict(color="crimson", width=3),
                  annotation_text=f"Price evaluated ({result['current_price']})")
    fig.update_layout(title=f"{commodity_2} — {result['verdict']}",
                       xaxis_title="Price", yaxis_title="Simulated frequency", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
with tab3:
    st.subheader("How much does sourcing-mix diversification reduce cost risk?")
    st.markdown(
        "Builds baskets that span from **1 cluster** (concentrated) to **many clusters** "
        "(diversified), holding basket size and per-item quantity constant, and compares the "
        "resulting cost volatility -- portfolio theory applied to a produce basket. Commodities "
        "within the same volatility cluster tend to move together (same weather sensitivity), "
        "so a basket spread across clusters smooths out: some items zig while others zag."
    )
    cluster_table, _ = compute_cluster_table()
    # restrict to commodities with enough daily history for a yearly STL fit -- otherwise
    # build_basket can pick a short-history commodity (e.g. Litchi(Local)) and the sweep
    # below crashes; same guard notebooks/04_business_simulations_clusters.ipynb applies.
    cluster_labels_series = cluster_table.loc[
        cluster_table.index.isin(commodities), "volatility_cluster_name"]

    c1, c2, c3 = st.columns(3)
    n_items = c1.slider("Basket size (commodities)", 3, 10, 6)
    quantity_each = c2.number_input("Quantity per commodity", min_value=1, value=100, step=10)
    n_sims_3 = c3.select_slider("Simulations per basket", [100, 200, 400], value=200)

    run = st.button("Run diversification sweep", type="primary")
    if run:
        cluster_labels_key = tuple(sorted(cluster_labels_series.dropna().items()))
        sweep_df = diversification_sweep(cluster_labels_key, n_items, quantity_each,
                                          horizon_days=90, n_sims=n_sims_3)
        if len(sweep_df) < 2:
            st.warning("Not enough distinct clusters/commodities available for this basket size.")
        else:
            reduction = (sweep_df["cv"].iloc[0] - sweep_df["cv"].iloc[-1]) / sweep_df["cv"].iloc[0]
            st.success(
                f"Spreading the same {n_items}-item basket from 1 cluster to "
                f"{int(sweep_df['n_clusters_used'].iloc[-1])} clusters reduces relative cost "
                f"risk (CV) by **{reduction:.0%}**, at essentially the same expected spend "
                f"({sweep_df['mean_cost'].iloc[0]:.0f} -> {sweep_df['mean_cost'].iloc[-1]:.0f})."
            )
            fig = go.Figure()
            fig.add_scatter(x=sweep_df["n_clusters_used"], y=sweep_df["cv"], mode="lines+markers",
                             line=dict(color="royalblue"))
            fig.update_layout(title=f"Diversification benefit — {n_items}-commodity basket",
                               xaxis_title="Number of distinct clusters sourced from",
                               yaxis_title="Basket cost CV (risk per unit of expected cost)",
                               height=450)
            st.plotly_chart(fig, use_container_width=True)

            concentrated = sweep_df.iloc[0]
            diversified = sweep_df.iloc[-1]
            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Concentrated basket (1 cluster)**")
                st.write(", ".join(concentrated["basket"]))
                st.caption(f"Mean cost {concentrated['mean_cost']:.0f}, CV {concentrated['cv']:.3f}")
            with colB:
                st.markdown(f"**Diversified basket ({int(diversified['n_clusters_used'])} clusters)**")
                st.write(", ".join(diversified["basket"]))
                st.caption(f"Mean cost {diversified['mean_cost']:.0f}, CV {diversified['cv']:.3f}")
    else:
        st.info("Set your basket parameters and click **Run diversification sweep** "
                "(runs several Monte Carlo simulations, takes a few seconds).")

# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Buy now, wait, or stay flexible?")
    st.markdown(
        "Compares three purchase strategies for acquiring a fixed quantity within a decision "
        "window: **buy now** at today's price, **wait** until a fixed future date, or **adaptive** "
        "-- buy as soon as price drops to a chosen threshold (or at the deadline if it never "
        "does). The adaptive strategy's value is genuine *optionality*, distinct from just the "
        "average forecast."
    )
    df_raw = load_raw_data()
    c1, c2, c3 = st.columns(3)
    commodity_4 = c1.selectbox("Commodity", commodities,
                                index=commodities.index(default_commodity), key="c4")
    quantity_4 = c2.number_input("Quantity to buy", min_value=1, value=500, step=50)
    decision_horizon = c3.slider("Decision horizon (days)", 30, 180, 90)

    c4, c5 = st.columns(2)
    fixed_wait_day = c4.slider("Fixed wait day (for the 'wait' strategy)", 1,
                                decision_horizon - 1, min(60, decision_horizon - 1))
    threshold_pct = c5.slider("Adaptive buy-threshold (% of today's price)", 80, 100, 95)

    current_price_4 = df_raw[df_raw[COMMODITY_COL] == commodity_4].sort_values(DATE_COL)["Average"].iloc[-1]
    paths_4, _, _ = simulate_price_paths_cached(commodity_4, horizon_days=max(decision_horizon, 91), n_sims=1000)
    comparison = sim.purchase_strategy_comparison(
        paths_4, current_price_4, quantity_4, fixed_wait_day=fixed_wait_day,
        threshold_frac=threshold_pct / 100, decision_horizon_days=decision_horizon)

    value_of_waiting = comparison.loc[0, "mean_cost"] - comparison.loc[1, "mean_cost"]
    value_of_optionality = comparison.loc[1, "mean_cost"] - comparison.loc[2, "mean_cost"]
    total_flex_value = comparison.loc[0, "mean_cost"] - comparison.loc[2, "mean_cost"]

    best_idx = comparison["mean_cost"].idxmin()
    st.success(f"Lowest expected cost: **{comparison.loc[best_idx, 'strategy']}** "
               f"(mean {comparison.loc[best_idx, 'mean_cost']:.0f})")
    st.caption(
        f"Value of waiting vs. buying now: {value_of_waiting:.0f}. "
        f"Additional value of adaptive vs. a fixed wait date: {value_of_optionality:.0f}. "
        f"Total value of flexibility: {total_flex_value:.0f}."
    )

    fig = go.Figure()
    fig.add_bar(x=comparison["strategy"], y=comparison["mean_cost"], marker_color="royalblue",
                error_y=dict(type="data", symmetric=False,
                              array=comparison["p90_cost"] - comparison["mean_cost"],
                              arrayminus=comparison["mean_cost"] - comparison["p10_cost"]))
    fig.update_layout(title=f"{commodity_4} — purchase strategy comparison",
                       yaxis_title="Total cost", height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comparison.round(2), use_container_width=True, hide_index=True)
