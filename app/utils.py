"""Shared data/model loading for the Streamlit app, wired to the pipelines and saved models
in src/ and models/ (see notebooks 03, 05, 04, 06 for where this logic comes from)."""

import json
import os
import sys

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import joblib
import pandas as pd
import streamlit as st

import data_pipeline
import forecasting_pipeline
import simulation_pipeline

CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "kalimati_tarkari_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CLUSTERING_CACHE_DIR = os.path.join(PROJECT_ROOT, "reports", "from notebooks", "clustering_outputs")
FORECASTING_CACHE_DIR = os.path.join(PROJECT_ROOT, "reports", "from notebooks", "forecasting_outputs")


@st.cache_data(show_spinner="Loading price data...")
def load_raw_data():
    return data_pipeline.load_and_clean(CSV_PATH)


@st.cache_data(show_spinner="Building weekly panel...")
def load_weekly_panel():
    return forecasting_pipeline.resample_weekly(load_raw_data())


# ---------------------------------------------------------------------------
# Clustering (notebook 03)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading clustering models...")
def load_cluster_pipelines():
    vol_pipe = joblib.load(os.path.join(MODELS_DIR, "volatility_cluster_pipeline.joblib"))
    season_pipe = joblib.load(os.path.join(MODELS_DIR, "seasonality_cluster_pipeline.joblib"))
    with open(os.path.join(MODELS_DIR, "cluster_metadata.json")) as f:
        meta = json.load(f)
    return vol_pipe, season_pipe, meta


@st.cache_data(show_spinner="Assigning commodities to clusters...")
def compute_cluster_table():
    """Feature tables come from the notebook's own cached output (same raw data, avoids
    re-running STL live); cluster labels are produced by the saved joblib pipelines."""
    vol = pd.read_csv(os.path.join(CLUSTERING_CACHE_DIR, "volatility_features.csv"), index_col="Commodity")
    seasonality = pd.read_csv(os.path.join(CLUSTERING_CACHE_DIR, "seasonality_features.csv"), index_col="Commodity")
    vol_pipe, season_pipe, meta = load_cluster_pipelines()

    vol_names = {int(k): v for k, v in meta["volatility_cluster_pipeline"]["cluster_names"].items()}
    season_names = {int(k): v for k, v in meta["seasonality_cluster_pipeline"]["cluster_names"].items()}

    features_a = data_pipeline.build_volatility_matrix(vol)
    vol_labels = vol_pipe.predict(features_a)
    table = vol.loc[features_a.index].copy()
    table["volatility_cluster"] = vol_labels
    table["volatility_cluster_name"] = [vol_names[label] for label in vol_labels]

    features_b = data_pipeline.build_seasonality_matrix(vol, seasonality)
    if len(features_b) > 0:
        season_labels = season_pipe.predict(features_b)
        season_lookup = pd.Series(
            [season_names[label] for label in season_labels], index=features_b.index)
        table["seasonality_cluster_name"] = table.index.map(season_lookup)
        strength_lookup = seasonality[["seasonal_strength", "trend_strength", "peak_month"]]
        table = table.join(strength_lookup)

    shape_path = os.path.join(CLUSTERING_CACHE_DIR, "commodity_clusters_named.csv")
    if os.path.exists(shape_path):
        named = pd.read_csv(shape_path, index_col=0)
        if "shape_cluster_name" in named.columns:
            table["shape_cluster_name"] = table.index.map(named["shape_cluster_name"])

    return table, meta


# ---------------------------------------------------------------------------
# Forecasting (notebook 05)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading forecasting model (TFT)...")
def load_forecast_model():
    from neuralforecast import NeuralForecast
    return NeuralForecast.load(path=os.path.join(MODELS_DIR, "forecast_final_model"))


@st.cache_data(show_spinner="Generating forecasts for all commodities...")
def compute_all_forecasts():
    weekly = load_weekly_panel()
    panel_df = forecasting_pipeline.build_nf_panel(weekly)
    nf = load_forecast_model()
    futr_df = forecasting_pipeline.add_calendar_exog(nf.make_future_dataframe(df=panel_df))
    forecast = nf.predict(df=panel_df, futr_df=futr_df)
    return forecast.rename(columns={"TFT": "forecast"})


@st.cache_data
def load_forecast_metadata():
    with open(os.path.join(MODELS_DIR, "forecast_metadata.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Business simulations (notebooks 04, 06)
# ---------------------------------------------------------------------------
STL_MIN_DAYS = int(365 * 1.5)


@st.cache_data(show_spinner=False)
def eligible_simulation_commodities():
    """Commodities with enough daily history (>= 1.5 years) for a yearly STL fit."""
    df = load_raw_data()
    counts = df.groupby(data_pipeline.COMMODITY_COL)[data_pipeline.DATE_COL].nunique()
    return sorted(counts[counts >= STL_MIN_DAYS].index.tolist())


@st.cache_data(show_spinner="Running Monte Carlo price simulation...")
def simulate_price_paths_cached(commodity, horizon_days=365, n_sims=500, block_size=14):
    df = load_raw_data()
    return simulation_pipeline.simulate_price_paths(
        df, commodity, horizon_days=horizon_days, n_sims=n_sims, block_size=block_size)


@st.cache_data(show_spinner="Checking today's price against typical seasonal pricing...")
def consumer_buy_timing_cached(commodity, current_price, current_date, horizon_days=90, n_sims=500):
    df = load_raw_data()
    return simulation_pipeline.consumer_buy_timing(
        df, commodity, current_price=current_price, current_date=current_date,
        horizon_days=horizon_days, n_sims=n_sims)


@st.cache_data(show_spinner="Simulating basket cost across diversification levels...")
def diversification_sweep(cluster_labels_key, n_items, quantity_each, horizon_days, n_sims, random_state=42):
    """cluster_labels_key: list of (commodity, cluster_name) pairs -- passed as a hashable tuple
    instead of a Series so Streamlit can cache on it."""
    cluster_labels = pd.Series(dict(cluster_labels_key))
    df = load_raw_data()
    max_clusters = cluster_labels.nunique()

    rows = []
    stl_cache = {}
    for n_clusters in range(1, max_clusters + 1):
        basket = simulation_pipeline.build_basket(cluster_labels, n_items, n_clusters, quantity_each, random_state)
        if len(basket) < n_items:
            continue
        paths = simulation_pipeline.simulate_basket_paths_correlated(
            df, list(basket.keys()), horizon_days=horizon_days, n_sims=n_sims, stl_cache=stl_cache)
        total_cost = sum(paths[c].mean(axis=1).values * qty for c, qty in basket.items())
        rows.append({
            "n_clusters_used": n_clusters,
            "mean_cost": total_cost.mean(),
            "std_cost": total_cost.std(),
            "cv": total_cost.std() / total_cost.mean(),
            "basket": list(basket.keys()),
        })
    return pd.DataFrame(rows)
