"""Monte Carlo price-simulation engine and business-decision logic, extracted from
notebooks/04_business_simulations_clusters.ipynb and notebooks/06_advanced_business_simulations.ipynb.

Core idea (shared by every function below): decompose a commodity's price into trend +
seasonal + residual via STL, extrapolate trend and seasonal forward, then block-bootstrap the
historical residuals thousands of times to get a distribution of plausible future price paths.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

DATE_COL = "Date"
COMMODITY_COL = "Commodity"

N_SIMS = 1000
HORIZON_DAYS = 365
BLOCK_SIZE = 14
TREND_LOOKBACK_DAYS = 180
RISK_AVERSION = 0.5


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    for col in ["Minimum", "Maximum", "Average"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[DATE_COL, COMMODITY_COL, "Average"])
    df = (df.groupby([COMMODITY_COL, DATE_COL], as_index=False)
            .agg({"Average": "mean"}))
    return df.sort_values([COMMODITY_COL, DATE_COL]).reset_index(drop=True)


def fit_stl(df, commodity, period=365):
    g = (df[df[COMMODITY_COL] == commodity]
         .set_index(DATE_COL)["Average"].sort_index().asfreq("D"))
    g = g.interpolate(limit=14).dropna()
    if len(g) < period * 1.5:
        raise ValueError(f"{commodity}: not enough history for a yearly STL fit "
                          f"({len(g)} days, need ~{int(period * 1.5)})")
    res = STL(g, period=period, robust=True).fit()
    return g, res


def _ols_slope(x, y):
    """Closed-form simple linear regression slope (degree-1 least squares).
    Equivalent to np.polyfit(x, y, 1)[0], but avoids np.polyfit's call into LAPACK's lstsq --
    which crashes when run off the main thread in this environment (e.g. inside Streamlit's
    script-runner thread) -- since this only needs elementwise numpy ops."""
    x_mean, y_mean = x.mean(), y.mean()
    x_dev = x - x_mean
    denom = (x_dev ** 2).sum()
    if denom == 0:
        return 0.0
    return (x_dev * (y - y_mean)).sum() / denom


def extrapolate_trend(trend, horizon_days, lookback_days=TREND_LOOKBACK_DAYS):
    recent = trend.iloc[-lookback_days:]
    x = np.arange(len(recent))
    slope = _ols_slope(x, recent.values)
    last_value = trend.iloc[-1]
    future_x = np.arange(1, horizon_days + 1)
    return last_value + slope * future_x


def extrapolate_seasonal(seasonal, future_dates):
    doy_avg = seasonal.groupby(seasonal.index.dayofyear).mean()
    future_doy = future_dates.dayofyear
    return np.array([doy_avg.get(d, doy_avg.get(365)) for d in future_doy])


def block_bootstrap_residuals(resid, horizon_days, block_size=BLOCK_SIZE, n_sims=N_SIMS):
    resid_vals = resid.values
    n = len(resid_vals)
    n_blocks_needed = int(np.ceil(horizon_days / block_size))
    sims = np.zeros((n_sims, horizon_days))
    for s in range(n_sims):
        blocks = []
        for _ in range(n_blocks_needed):
            start = np.random.randint(0, n - block_size)
            blocks.append(resid_vals[start:start + block_size])
        sims[s] = np.concatenate(blocks)[:horizon_days]
    return sims


def simulate_price_paths(df, commodity, horizon_days=HORIZON_DAYS, n_sims=N_SIMS, block_size=BLOCK_SIZE):
    """Returns (price_paths DataFrame [n_sims x horizon_days], historical series, STL result)."""
    g, res = fit_stl(df, commodity)
    last_date = g.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    trend_future = extrapolate_trend(res.trend, horizon_days)
    seasonal_future = extrapolate_seasonal(res.seasonal, future_dates)
    resid_sims = block_bootstrap_residuals(res.resid, horizon_days, block_size, n_sims)

    price_paths = trend_future[None, :] + seasonal_future[None, :] + resid_sims
    price_paths = np.clip(price_paths, a_min=0, a_max=None)
    return pd.DataFrame(price_paths, columns=future_dates), g, res


def var_cvar(costs, confidence=0.95):
    var = np.percentile(costs, confidence * 100)
    cvar = costs[costs >= var].mean()
    return var, cvar


# ---------------------------------------------------------------------------
# Section 3 (notebook 04): Farmer selling-window recommendation
# ---------------------------------------------------------------------------
def farmer_selling_window(paths, risk_aversion=RISK_AVERSION):
    """Best future calendar month to sell, trading expected price against downside variance."""
    future_dates = paths.columns
    month_labels = future_dates.month

    records = []
    for m in sorted(set(month_labels)):
        cols = future_dates[month_labels == m]
        month_prices = paths[cols].values.flatten()
        records.append({
            "month": m,
            "expected_price": month_prices.mean(),
            "std_price": month_prices.std(),
            "p10": np.percentile(month_prices, 10),
            "p90": np.percentile(month_prices, 90),
            "cv": month_prices.std() / month_prices.mean(),
        })
    result = pd.DataFrame(records)
    result["risk_adjusted_score"] = result["expected_price"] - risk_aversion * result["std_price"]
    result = result.sort_values("risk_adjusted_score", ascending=False).reset_index(drop=True)
    result["rank"] = result.index + 1
    return result


# ---------------------------------------------------------------------------
# Section 6 (notebook 04): Consumer "good time to buy" indicator
# ---------------------------------------------------------------------------
def consumer_buy_timing(df, commodity, current_price=None, current_date=None,
                         horizon_days=90, n_sims=N_SIMS):
    g, res = fit_stl(df, commodity)

    if current_price is None or current_date is None:
        current_price = g.iloc[-1]
        current_date = g.index[-1]

    target_month = pd.Timestamp(current_date).month
    paths, _, _ = simulate_price_paths(df, commodity, horizon_days=horizon_days, n_sims=n_sims)
    future_dates = paths.columns
    cols = future_dates[future_dates.month == target_month]
    if len(cols) == 0:
        paths, _, _ = simulate_price_paths(df, commodity, horizon_days=400, n_sims=n_sims)
        future_dates = paths.columns
        cols = future_dates[future_dates.month == target_month]

    typical_dist = paths[cols].values.flatten()
    cheaper_than_pct = (typical_dist > current_price).mean() * 100

    if cheaper_than_pct >= 70:
        verdict = "Good time to buy"
    elif cheaper_than_pct <= 30:
        verdict = "Expensive right now"
    else:
        verdict = "Roughly average pricing"

    month_name = pd.Timestamp(current_date).strftime("%B")
    return {
        "commodity": commodity,
        "current_price": round(float(current_price), 2),
        "current_date": pd.Timestamp(current_date).date(),
        "month": month_name,
        "typical_median": round(float(np.median(typical_dist)), 2),
        "typical_p10": round(float(np.percentile(typical_dist, 10)), 2),
        "typical_p90": round(float(np.percentile(typical_dist, 90)), 2),
        "cheaper_than_pct": round(cheaper_than_pct, 1),
        "verdict": verdict,
        "message": (f"Today's {commodity} price ({current_price:.2f}) is cheaper than "
                    f"{cheaper_than_pct:.0f}% of typical {month_name} prices — {verdict.lower()}."),
        "typical_dist": typical_dist,
    }


# ---------------------------------------------------------------------------
# Section 4/5 (notebook 04): correlated basket simulation + cluster diversification
# ---------------------------------------------------------------------------
def build_common_residuals(df, commodities, stl_cache=None):
    fits = {}
    resid_series = {}
    for c in commodities:
        if stl_cache is not None and c in stl_cache:
            g, res = stl_cache[c]
        else:
            g, res = fit_stl(df, c)
            if stl_cache is not None:
                stl_cache[c] = (g, res)
        fits[c] = (g, res)
        resid_series[c] = res.resid
    resid_df = pd.DataFrame(resid_series).dropna()
    return fits, resid_df


def simulate_basket_paths_correlated(df, commodities, horizon_days=90, n_sims=500,
                                      block_size=BLOCK_SIZE, stl_cache=None):
    """Bootstraps the SAME historical calendar blocks across every commodity at once, so real
    co-movement (e.g. a bad week hitting several vegetables together) is preserved -- unlike
    simulating each commodity independently, which overstates diversification."""
    fits, resid_df = build_common_residuals(df, commodities, stl_cache)
    n_common_days = len(resid_df)
    n_blocks_needed = int(np.ceil(horizon_days / block_size))
    last_date = max(g.index[-1] for g, _ in fits.values())
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    trend_future = {c: extrapolate_trend(res.trend, horizon_days) for c, (g, res) in fits.items()}
    seasonal_future = {c: extrapolate_seasonal(res.seasonal, future_dates) for c, (g, res) in fits.items()}

    basket_paths = {c: np.zeros((n_sims, horizon_days)) for c in commodities}
    for s in range(n_sims):
        block_starts = np.random.randint(0, n_common_days - block_size, size=n_blocks_needed)
        resid_blocks = np.concatenate([resid_df.values[st:st + block_size] for st in block_starts], axis=0)
        resid_blocks = resid_blocks[:horizon_days]
        for i, c in enumerate(commodities):
            path = trend_future[c] + seasonal_future[c] + resid_blocks[:, i]
            basket_paths[c][s] = np.clip(path, 0, None)

    return {c: pd.DataFrame(basket_paths[c], columns=future_dates) for c in commodities}


def build_basket(cluster_labels, n_items, n_clusters_to_use, quantity_each=100, random_state=42):
    """Round-robins through n_clusters_to_use distinct clusters, one new commodity from each in
    turn, until n_items total are chosen. n_clusters_to_use=1 gives a fully concentrated basket."""
    rng = np.random.RandomState(random_state)
    available_clusters = cluster_labels.unique().tolist()
    rng.shuffle(available_clusters)
    chosen_clusters = available_clusters[:n_clusters_to_use]

    pools = {c: rng.permutation(cluster_labels[cluster_labels == c].index.values).tolist()
             for c in chosen_clusters}

    basket = {}
    i = 0
    while len(basket) < n_items:
        cluster = chosen_clusters[i % len(chosen_clusters)]
        if pools[cluster]:
            commodity = pools[cluster].pop()
            if commodity not in basket:
                basket[commodity] = quantity_each
        i += 1
        if i > 10_000:
            break
    return basket


# ---------------------------------------------------------------------------
# Section 3 (notebook 06): Buy-now vs. wait-and-see (real options)
# ---------------------------------------------------------------------------
def adaptive_purchase_cost(paths, threshold, quantity):
    costs = np.zeros(len(paths))
    for i, row in enumerate(paths.values):
        below = np.where(row <= threshold)[0]
        buy_day = below[0] if len(below) > 0 else len(row) - 1
        costs[i] = row[buy_day] * quantity
    return costs


def purchase_strategy_comparison(paths, current_price, quantity, fixed_wait_day=60,
                                  threshold_frac=0.95, decision_horizon_days=90):
    decision_paths = paths.iloc[:, :decision_horizon_days]
    n_sims = len(paths)

    cost_buy_now = np.full(n_sims, current_price * quantity)
    cost_wait_fixed = decision_paths.iloc[:, min(fixed_wait_day, decision_horizon_days - 1)].values * quantity
    threshold_price = current_price * threshold_frac
    cost_adaptive = adaptive_purchase_cost(decision_paths, threshold_price, quantity)

    return pd.DataFrame({
        "strategy": ["Buy Now", f"Wait until day {fixed_wait_day}", "Adaptive (real option)"],
        "mean_cost": [cost_buy_now.mean(), cost_wait_fixed.mean(), cost_adaptive.mean()],
        "std_cost": [cost_buy_now.std(), cost_wait_fixed.std(), cost_adaptive.std()],
        "p10_cost": [np.percentile(cost_buy_now, 10), np.percentile(cost_wait_fixed, 10),
                     np.percentile(cost_adaptive, 10)],
        "p90_cost": [np.percentile(cost_buy_now, 90), np.percentile(cost_wait_fixed, 90),
                     np.percentile(cost_adaptive, 90)],
    })
