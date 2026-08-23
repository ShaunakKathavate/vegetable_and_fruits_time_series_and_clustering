"""Fit and persist the two commodity-clustering models from notebooks/03_commodity_clustering.ipynb.

Run from the project root:
    python src/train_clustering_models.py

Writes to models/:
    volatility_cluster_pipeline.joblib   -- StandardScaler + KMeans, fit on ALL commodities
    seasonality_cluster_pipeline.joblib  -- StandardScaler + KMeans, fit on Tier-1 commodities only
    cluster_metadata.json                -- feature order, cluster-name labels, config thresholds
"""

import argparse
import json
import os

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

# scikit-learn's compiled KMeans (sklearn.cluster.KMeans) crashes on this machine's conda
# env -- its Cython/OpenMP internals hit a broken vcomp/OpenBLAS DLL (native crash, exit code
# 0xc06d007f, reproducible even with threadpoolctl fully stubbed out). SimpleKMeans is a
# pure-NumPy stand-in with the same fit/predict surface, unaffected by that.
from simple_kmeans import SimpleKMeans as KMeans

from data_pipeline import (
    TIER1_MIN_OBS,
    TIER1_MIN_SPAN_DAYS,
    VOLATILITY_FEATURE_COLS,
    SEASONALITY_FEATURE_COLS,
    build_seasonality_matrix,
    build_volatility_matrix,
    compute_coverage_tiers,
    compute_seasonality_features,
    compute_volatility_features,
    load_and_clean,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "kalimati_tarkari_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
# Feature tables the notebook itself already computed from this same raw CSV (see
# notebooks/03_commodity_clustering.ipynb section 2-3). Reusing them skips the slow STL
# refit (minutes) -- pass --recompute to rebuild them from raw data instead.
CACHE_DIR = os.path.join(PROJECT_ROOT, "reports", "from notebooks", "clustering_outputs")
VOLATILITY_CACHE = os.path.join(CACHE_DIR, "volatility_features.csv")
SEASONALITY_CACHE = os.path.join(CACHE_DIR, "seasonality_features.csv")

# k chosen in the notebook via silhouette search (both feature sets picked k=2)
K_VOLATILITY = 2
K_SEASONALITY = 2
RANDOM_STATE = 42


def volatility_label(rank, total):
    frac = rank / max(total - 1, 1)
    if frac < 0.34:
        return "Stable / Low Volatility"
    elif frac < 0.67:
        return "Moderate Volatility"
    return "High Volatility"


def full_label(cv, seasonal_strength, cv_median, season_median):
    vol_tag = "High-Volatility" if cv >= cv_median else "Stable"
    season_tag = "Strongly Seasonal" if seasonal_strength >= season_median else "Weakly Seasonal"
    return f"{vol_tag}, {season_tag}"


def compute_features_from_raw():
    print(f"Loading raw data from {CSV_PATH} ...")
    df = load_and_clean(CSV_PATH)
    coverage = compute_coverage_tiers(df)
    vol = compute_volatility_features(df, coverage)

    tier1_commodities = coverage[coverage["tier"] == "Tier1_full_history"].index.tolist()
    print(f"Running STL for {len(tier1_commodities)} Tier-1 commodities (this is the slow step)...")
    seasonality = compute_seasonality_features(df, tier1_commodities)
    return vol, seasonality


def load_features_from_cache():
    print(f"Loading precomputed feature tables from {CACHE_DIR} ...")
    vol = pd.read_csv(VOLATILITY_CACHE, index_col="Commodity")
    seasonality = pd.read_csv(SEASONALITY_CACHE, index_col="Commodity")
    return vol, seasonality


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true",
                         help="Recompute features from raw data instead of using the cached CSVs")
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)

    use_cache = not args.recompute and os.path.exists(VOLATILITY_CACHE) and os.path.exists(SEASONALITY_CACHE)
    vol, seasonality = load_features_from_cache() if use_cache else compute_features_from_raw()

    # --- Volatility-only pipeline (all commodities) ---
    features_a = build_volatility_matrix(vol)
    pipeline_a = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=K_VOLATILITY, n_init=10, random_state=RANDOM_STATE)),
    ])
    labels_a = pipeline_a.fit_predict(features_a)
    joblib.dump(pipeline_a, os.path.join(MODELS_DIR, "volatility_cluster_pipeline.joblib"))
    print(f"Saved volatility_cluster_pipeline.joblib (k={K_VOLATILITY}, n={len(features_a)})")

    stats_a = features_a.assign(cluster=labels_a).groupby("cluster")["cv"].mean().sort_values()
    volatility_names = {
        int(cluster): volatility_label(rank, len(stats_a))
        for rank, cluster in enumerate(stats_a.index)
    }

    # --- Volatility + seasonality pipeline (Tier 1 only) ---
    features_b = build_seasonality_matrix(vol, seasonality)
    pipeline_b = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=K_SEASONALITY, n_init=10, random_state=RANDOM_STATE)),
    ])
    labels_b = pipeline_b.fit_predict(features_b)
    joblib.dump(pipeline_b, os.path.join(MODELS_DIR, "seasonality_cluster_pipeline.joblib"))
    print(f"Saved seasonality_cluster_pipeline.joblib (k={K_SEASONALITY}, n={len(features_b)})")

    stats_b = features_b.assign(cluster=labels_b).groupby("cluster")[["cv", "seasonal_strength"]].mean()
    cv_median = stats_b["cv"].median()
    season_median = stats_b["seasonal_strength"].median()
    seasonality_names = {
        int(cluster): full_label(row["cv"], row["seasonal_strength"], cv_median, season_median)
        for cluster, row in stats_b.iterrows()
    }

    metadata = {
        "config": {
            "tier1_min_span_days": TIER1_MIN_SPAN_DAYS,
            "tier1_min_obs": TIER1_MIN_OBS,
            "random_state": RANDOM_STATE,
        },
        "volatility_cluster_pipeline": {
            "feature_columns": VOLATILITY_FEATURE_COLS,
            "applies_to": "all commodities",
            "n_clusters": K_VOLATILITY,
            "cluster_names": volatility_names,
        },
        "seasonality_cluster_pipeline": {
            "feature_columns": SEASONALITY_FEATURE_COLS,
            "applies_to": "Tier1_full_history commodities only",
            "n_clusters": K_SEASONALITY,
            "cluster_names": seasonality_names,
        },
    }
    with open(os.path.join(MODELS_DIR, "cluster_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved cluster_metadata.json")


if __name__ == "__main__":
    main()
