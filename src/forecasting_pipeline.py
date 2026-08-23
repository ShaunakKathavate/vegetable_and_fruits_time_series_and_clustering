"""Reusable data pipeline for the forecasting-backtest notebook (notebooks/05_forecasting_backtest.ipynb).

Raw daily prices -> weekly panel per commodity -> eligibility filter -> long-format panel
(unique_id/ds/y + calendar exogenous features) ready for the neuralforecast models.
"""

import pandas as pd

DATE_COL = "Date"
COMMODITY_COL = "Commodity"

# Same thresholds as the notebook's backtest config -- used only to decide which commodities
# have enough history to be backtested at all, not by the final single-model training itself.
HORIZON_WEEKS = 8
N_ORIGINS = 4
MIN_TRAIN_WEEKS = 104

NF_EXOG_COLS = ["month", "weekofyear"]

# Final-forecast config (notebook section 11): a bigger, one-off training budget, trained
# globally across every commodity rather than on a single ~300-row series in isolation.
NF_FINAL_MAX_STEPS = 100
NF_FINAL_INPUT_SIZE = 52
NF_FINAL_TFT_HIDDEN_SIZE = 32
NF_FINAL_TFT_N_HEAD = 2
FINAL_HORIZON_WEEKS = 12


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["Average"] = pd.to_numeric(df["Average"], errors="coerce")
    df = df.dropna(subset=[DATE_COL, COMMODITY_COL, "Average"])
    df = (df.groupby([COMMODITY_COL, DATE_COL], as_index=False).agg({"Average": "mean"}))
    return df.sort_values([COMMODITY_COL, DATE_COL]).reset_index(drop=True)


def resample_weekly(df):
    """Long-format weekly panel: one row per (Commodity, week). Bridges gaps of <=2 weeks."""
    weekly = (df.set_index(DATE_COL)
                .groupby(COMMODITY_COL)["Average"]
                .resample("W")
                .mean()
                .rename("avg_price")
                .reset_index())
    weekly["avg_price"] = weekly.groupby(COMMODITY_COL)["avg_price"].transform(
        lambda s: s.interpolate(limit=2))
    return weekly.dropna(subset=["avg_price"])


def compute_eligible_commodities(weekly, min_train_weeks=MIN_TRAIN_WEEKS,
                                  n_origins=N_ORIGINS, horizon_weeks=HORIZON_WEEKS):
    """Commodities with enough weekly history to run a rolling-origin backtest at all."""
    required_weeks = min_train_weeks + n_origins * horizon_weeks
    week_counts = weekly.groupby(COMMODITY_COL)[DATE_COL].count()
    return week_counts[week_counts >= required_weeks].index.tolist()


def add_calendar_exog(df):
    """Attach month / week-of-year columns (expects a 'ds' date column) -- used as both
    history and known-future inputs to the neuralforecast models."""
    df = df.copy()
    df["month"] = df["ds"].dt.month
    df["weekofyear"] = df["ds"].dt.isocalendar().week.astype(int)
    return df


def build_nf_panel(weekly):
    """weekly (Commodity, Date, avg_price) -> long panel (unique_id, ds, y, month, weekofyear)
    in the format neuralforecast expects, with every commodity's history included."""
    panel = weekly[[COMMODITY_COL, DATE_COL, "avg_price"]].rename(
        columns={COMMODITY_COL: "unique_id", DATE_COL: "ds", "avg_price": "y"})
    return add_calendar_exog(panel)
