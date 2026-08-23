"""Reusable data pipeline for the commodity clustering notebook (notebooks/03_commodity_clustering.ipynb).

Turns the raw Kalimati Tarkari CSV into the feature tables the clustering models are trained on:
raw prices -> clean/dedupe -> tier by history length -> volatility features (all commodities)
                                                      -> seasonality features (Tier 1 only, via STL)
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

DATE_COL = "Date"
COMMODITY_COL = "Commodity"

# Tier-1 eligibility: enough span + enough observations to trust a yearly STL decomposition
TIER1_MIN_SPAN_DAYS = 730     # ~2 years
TIER1_MIN_OBS = 200

# Minimum months of coverage required to build a seasonal (month-of-year) profile at all
MIN_MONTHS_FOR_PROFILE = 6

VOLATILITY_FEATURE_COLS = ["cv", "avg_daily_range_pct", "log_price_mean"]
SEASONALITY_FEATURE_COLS = ["cv", "avg_daily_range_pct", "log_price_mean", "seasonal_strength", "trend_strength"]


def load_and_clean(csv_path):
    """Raw CSV -> one row per (Commodity, Date), numeric price columns, sorted."""
    df = pd.read_csv(csv_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    for col in ["Minimum", "Maximum", "Average"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[DATE_COL, COMMODITY_COL, "Average"])

    df = (df.groupby([COMMODITY_COL, DATE_COL], as_index=False)
            .agg({"Unit": "first", "Minimum": "mean", "Maximum": "mean", "Average": "mean"}))
    df = df.sort_values([COMMODITY_COL, DATE_COL]).reset_index(drop=True)

    df["Price_Range_Pct"] = (df["Maximum"] - df["Minimum"]) / df["Average"].replace(0, np.nan)
    df["Year"] = df[DATE_COL].dt.year
    df["Month"] = df[DATE_COL].dt.month
    return df


def compute_coverage_tiers(df, min_span_days=TIER1_MIN_SPAN_DAYS, min_obs=TIER1_MIN_OBS):
    """Per-commodity history length -> Tier1_full_history / Tier2_sparse."""
    rows = []
    for name, g in df.groupby(COMMODITY_COL):
        span_days = (g[DATE_COL].max() - g[DATE_COL].min()).days + 1
        months_covered = (g["Month"].nunique() + 12 * (g["Year"].nunique() - 1)
                           if g["Year"].nunique() else g["Month"].nunique())
        rows.append({"Commodity": name, "n_obs": len(g), "span_days": span_days,
                      "months_covered": months_covered})
    coverage = pd.DataFrame(rows)
    coverage["tier"] = np.where(
        (coverage["span_days"] >= min_span_days) & (coverage["n_obs"] >= min_obs),
        "Tier1_full_history", "Tier2_sparse")
    return coverage.set_index("Commodity")


def compute_volatility_features(df, coverage):
    """Volatility features for every commodity, joined with tier/coverage info."""
    vol = df.groupby(COMMODITY_COL)["Average"].agg(["mean", "std"]).rename(
        columns={"mean": "price_mean", "std": "price_std"})
    vol["cv"] = vol["price_std"] / vol["price_mean"]
    vol["avg_daily_range_pct"] = df.groupby(COMMODITY_COL)["Price_Range_Pct"].mean()
    vol["log_price_mean"] = np.log1p(vol["price_mean"])
    vol = vol.merge(coverage[["tier", "n_obs", "span_days"]], left_index=True, right_index=True)
    return vol


def _stl_features(df, commodity, period=365):
    g = (df[df[COMMODITY_COL] == commodity]
         .set_index(DATE_COL)["Average"].sort_index().asfreq("D"))
    g = g.interpolate(limit=14).dropna()
    if len(g) < period * 1.5:
        return None
    res = STL(g, period=period, robust=True).fit()
    seasonal_strength = max(0, 1 - res.resid.var() / (res.seasonal + res.resid).var())
    trend_strength = max(0, 1 - res.resid.var() / (res.trend + res.resid).var())
    peak_month = res.seasonal.groupby(res.seasonal.index.month).mean().idxmax()
    return {"Commodity": commodity, "seasonal_strength": round(seasonal_strength, 3),
            "trend_strength": round(trend_strength, 3), "peak_month": peak_month}


def compute_seasonality_features(df, tier1_commodities, period=365):
    """STL seasonal/trend strength + peak month, for Tier-1 commodities only."""
    rows = [r for r in (_stl_features(df, c, period=period) for c in tier1_commodities) if r]
    return pd.DataFrame(rows).set_index("Commodity")


def build_volatility_matrix(vol):
    """Feature matrix (all commodities) for the volatility-only clustering model."""
    return vol[VOLATILITY_FEATURE_COLS].dropna()


def build_seasonality_matrix(vol, seasonality):
    """Feature matrix (Tier-1 only) for the volatility+seasonality clustering model."""
    return (vol.loc[vol["tier"] == "Tier1_full_history", VOLATILITY_FEATURE_COLS]
               .join(seasonality[["seasonal_strength", "trend_strength"]], how="inner"))
