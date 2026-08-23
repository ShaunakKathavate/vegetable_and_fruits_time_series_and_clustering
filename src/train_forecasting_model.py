"""Fit and persist the "final forecast" model from notebooks/05_forecasting_backtest.ipynb (section 11).

Run from the project root:
    python src/train_forecasting_model.py

The notebook picks its deployable model by rolling-origin backtesting {SARIMA, Prophet, LSTM, TFT}
across a sample of commodities and taking whichever has the lowest average MAPE (see
reports/from notebooks/forecasting_outputs/backtest_all_commodities.csv, produced by a prior run
of the notebook's sections 9-10 against this same raw CSV). Re-running that full backtest here
would refit dozens of models from scratch; this script reuses that cached ranking (pass
--recompute-backtest to instead redo it) and only runs the expensive one-off step notebooks/05
itself calls "final": training the winning model type on the full weekly panel and saving it.

Writes to models/:
    forecast_final_model/            -- neuralforecast checkpoint (if TFT or LSTM won), OR
    forecast_final_model.pickle      -- statsmodels SARIMAXResults (if SARIMA won), OR
    forecast_final_model.json        -- Prophet model (if Prophet won)
    forecast_metadata.json           -- which model won and why, target commodity, config, forecast
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

from forecasting_pipeline import (
    FINAL_HORIZON_WEEKS,
    NF_EXOG_COLS,
    NF_FINAL_INPUT_SIZE,
    NF_FINAL_MAX_STEPS,
    NF_FINAL_TFT_HIDDEN_SIZE,
    NF_FINAL_TFT_N_HEAD,
    add_calendar_exog,
    build_nf_panel,
    compute_eligible_commodities,
    load_and_clean,
    resample_weekly,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "kalimati_tarkari_dataset.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BACKTEST_CACHE = os.path.join(PROJECT_ROOT, "reports", "from notebooks", "forecasting_outputs",
                               "backtest_all_commodities.csv")

CANDIDATE_MODELS = ["SARIMA", "Prophet", "LSTM", "TFT"]
BASELINE_MODELS = ["Naive Seasonal", "Moving Average"]


def moving_average_forecast(train, horizon, window=8):
    return np.repeat(train.iloc[-window:].mean(), horizon)


def sarima_forecast(train, horizon, order=(1, 1, 1), seasonal_order=(1, 0, 1, 52)):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    model = SARIMAX(train.values, order=order, seasonal_order=seasonal_order,
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    lo, hi = train.min(), train.max()
    pad = max(hi - lo, hi * 0.5, 1.0)
    pred = np.clip(fit.forecast(steps=horizon), lo - pad, hi + pad)
    return fit, pred


def prophet_forecast(train, horizon):
    from prophet import Prophet
    prophet_df = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=horizon, freq="W", include_history=False)
    pred = m.predict(future)["yhat"].values
    return m, pred


def neuralforecast_final(model_cls, panel_df, target_commodity, horizon, **model_kwargs):
    from neuralforecast import NeuralForecast
    model = model_cls(h=horizon, input_size=NF_FINAL_INPUT_SIZE, max_steps=NF_FINAL_MAX_STEPS,
                       hist_exog_list=NF_EXOG_COLS, futr_exog_list=NF_EXOG_COLS,
                       start_padding_enabled=True,
                       enable_progress_bar=False, accelerator="cpu", logger=False,
                       enable_checkpointing=False, val_check_steps=10_000,
                       early_stop_patience_steps=-1, random_seed=42, **model_kwargs)
    nf = NeuralForecast(models=[model], freq="W")
    nf.fit(df=panel_df)
    futr_df = add_calendar_exog(nf.make_future_dataframe())
    forecast = nf.predict(futr_df=futr_df)
    target_forecast = forecast[forecast["unique_id"] == target_commodity].sort_values("ds")
    return nf, target_forecast[model_cls.__name__].values


def pick_final_model_from_cache():
    df = pd.read_csv(BACKTEST_CACHE)
    overall_mape = df.groupby("model")["mape"].mean().sort_values()
    candidate_ranking = overall_mape[overall_mape.index.isin(CANDIDATE_MODELS)].dropna()
    baseline_ranking = overall_mape[overall_mape.index.isin(BASELINE_MODELS)].dropna()
    return candidate_ranking, baseline_ranking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute-backtest", action="store_true",
                         help="Redo the full multi-model backtest instead of using the cached ranking")
    args = parser.parse_args()
    if args.recompute_backtest:
        raise NotImplementedError(
            "Re-running the full SARIMA/Prophet/LSTM/TFT backtest is not wired up in this script -- "
            "run notebooks/05_forecasting_backtest.ipynb sections 9-10 to refresh "
            f"{BACKTEST_CACHE}, then re-run this script without --recompute-backtest.")

    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Loading raw data from {CSV_PATH} ...")
    df = load_and_clean(CSV_PATH)
    weekly = resample_weekly(df)
    eligible_commodities = compute_eligible_commodities(weekly)
    demo_commodity = eligible_commodities[0]
    print(f"Target commodity: {demo_commodity} ({len(eligible_commodities)} eligible commodities total)")

    candidate_ranking, baseline_ranking = pick_final_model_from_cache()
    final_model_name = candidate_ranking.index[0]
    beats_baseline = bool(candidate_ranking.iloc[0] < baseline_ranking.iloc[0])
    print(f"Final model (from cached backtest ranking): {final_model_name} "
          f"(MAPE {candidate_ranking.iloc[0]:.1f}% vs strongest baseline "
          f"{baseline_ranking.index[0]} {baseline_ranking.iloc[0]:.1f}%)")

    full_series = (weekly[weekly["Commodity"] == demo_commodity]
                   .set_index("Date")["avg_price"].sort_index())
    last_date = full_series.index[-1]
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=FINAL_HORIZON_WEEKS, freq="W")

    artifact_path = None
    if final_model_name == "SARIMA":
        fitted, final_pred = sarima_forecast(full_series, FINAL_HORIZON_WEEKS)
        artifact_path = os.path.join(MODELS_DIR, "forecast_final_model.pickle")
        fitted.save(artifact_path)
    elif final_model_name == "Prophet":
        from prophet.serialize import model_to_json
        fitted, final_pred = prophet_forecast(full_series, FINAL_HORIZON_WEEKS)
        artifact_path = os.path.join(MODELS_DIR, "forecast_final_model.json")
        with open(artifact_path, "w") as f:
            f.write(model_to_json(fitted))
    elif final_model_name in ("LSTM", "TFT"):
        from neuralforecast.models import LSTM, TFT
        panel_df = build_nf_panel(weekly)
        model_cls = TFT if final_model_name == "TFT" else LSTM
        extra_kwargs = ({"hidden_size": NF_FINAL_TFT_HIDDEN_SIZE, "n_head": NF_FINAL_TFT_N_HEAD}
                         if final_model_name == "TFT" else {})
        print(f"Training {final_model_name} globally across {panel_df['unique_id'].nunique()} "
              f"commodities ({NF_FINAL_MAX_STEPS} steps)... this is the slow step.")
        nf, final_pred = neuralforecast_final(model_cls, panel_df, demo_commodity,
                                               FINAL_HORIZON_WEEKS, **extra_kwargs)
        artifact_path = os.path.join(MODELS_DIR, "forecast_final_model")
        nf.save(path=artifact_path, save_dataset=False, overwrite=True)
    else:
        raise RuntimeError(f"No save path implemented for final_model_name={final_model_name!r}")

    print(f"Saved {final_model_name} model -> {artifact_path}")

    metadata = {
        "final_model": final_model_name,
        "target_commodity": demo_commodity,
        "beats_strongest_baseline": beats_baseline,
        "candidate_backtest_mape": candidate_ranking.round(2).to_dict(),
        "baseline_backtest_mape": baseline_ranking.round(2).to_dict(),
        "backtest_cache_source": os.path.relpath(BACKTEST_CACHE, PROJECT_ROOT),
        "final_horizon_weeks": FINAL_HORIZON_WEEKS,
        "artifact_path": os.path.relpath(artifact_path, PROJECT_ROOT),
        "forecast": [
            {"date": str(d.date()), "forecast": round(float(p), 2)}
            for d, p in zip(future_dates, final_pred)
        ],
    }
    with open(os.path.join(MODELS_DIR, "forecast_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved forecast_metadata.json")


if __name__ == "__main__":
    main()
